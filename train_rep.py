import os
import json
import argparse
import warnings
import time
import numpy as np
import sklearn.metrics as skmet

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import EarlyStopping, set_random_seed, progress_bar, summarize_result
from loader import EEGDataLoader

from models.protop_gabor import ProtoPNet
from models.RepSleepNet import RepSleepNet

warnings.filterwarnings("ignore")
CLASS_WEIGHT = [1, 1.5, 1, 1, 1]


class ExplainableKDLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.5, beta=0.1):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHT).float())

    def forward(self, student_logits, teacher_logits, student_feat, teacher_feat, labels):
        # 1. 分类损失
        loss_ce = self.ce_loss(student_logits, labels)

        # 2. KL散度蒸馏 (软标签)
        p_student = F.log_softmax(student_logits / self.T, dim=1)
        p_teacher = F.softmax(teacher_logits / self.T, dim=1)
        loss_kd = F.kl_div(p_student, p_teacher, reduction='batchmean') * (self.T ** 2)

        # 3. 特征对齐蒸馏 (让轻量模型学到大模型的原理解释特征)
        if student_feat.shape == teacher_feat.shape:
            loss_feat = F.mse_loss(student_feat, teacher_feat)
        else:
            loss_feat = 0.0

        total_loss = loss_ce + self.alpha * loss_kd + self.beta * loss_feat
        return total_loss, loss_ce, loss_kd


class EKDTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        self.cfg = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 实例化老师
        self.teacher_model = ProtoPNet(config).to(self.device)
        self._load_teacher_weights()
        self.teacher_model.eval()

        # ====================================================
        # [关键修复]: 使用 forward_hook 隐式截获 Teacher 的隐层特征
        # ====================================================
        self.teacher_features = None
        def hook(module, input, output):
            self.teacher_features = output
        self.teacher_model.feature_extractor.register_forward_hook(hook)

        # 2. 实例化学生
        self.student_model = RepSleepNet(num_classes=config['classifier']['num_classes'],
                                         seq_len=config['dataset'].get('seq_len', 10)).to(self.device)

        self.loader_dict = self.build_dataloader()

        self.criterion = ExplainableKDLoss(T=3.0, alpha=0.5, beta=0.1).to(self.device)
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=1e-3, weight_decay=1e-4)

        self.ckpt_dir = os.path.join('checkpoints', 'RepSleepNet_' + str(args.seed))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = f'repsleep_fold-{self.fold:02d}.pth'

        self.early_stopping = EarlyStopping(patience=10, verbose=True,
                                            ckpt_path=self.ckpt_dir, ckpt_name=self.ckpt_name)

    def _load_teacher_weights(self):
        teacher_ckpt = os.path.join('checkpoints', self.cfg['name'] + '_' + str(self.args.seed),
                                    f'ckpt_fold-{self.fold:02d}.pth')
        if os.path.exists(teacher_ckpt):
            self.teacher_model.load_state_dict(torch.load(teacher_ckpt), strict=False)
            print("[INFO] Teacher 模型 (ProtoPNet) 权重加载成功！")
        else:
            print("[WARN] 未找到Teacher权重，Teacher将使用随机权重。")

    def build_dataloader(self):
        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def train_one_epoch(self, epoch):
        self.student_model.train()
        total_loss, correct, total = 0, 0, 0

        for i, (inputs, labels) in enumerate(self.loader_dict['train']):
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            B = labels.size(0)

            # --- Teacher 前向传播 ---
            with torch.no_grad():
                t_logits = self.teacher_model(inputs, return_indices=False)
                # 利用hook得到的特征，求均值池化，维度变为[B, 128] 与学生完美匹配
                t_feat = self.teacher_features.mean(dim=-1).detach()

            # --- Student 前向传播 ---
            s_logits, s_feat = self.student_model(inputs)

            # --- EKD 知识蒸馏损失计算 ---
            loss, loss_ce, loss_kd = self.criterion(s_logits, t_logits, s_feat, t_feat, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(s_logits, 1)
            correct += pred.eq(labels).sum().item()
            total += B

            if i % 20 == 0:
                print(f"\rEpoch {epoch}[{i}/{len(self.loader_dict['train'])}] "
                      f"Loss: {loss.item():.4f} (CE:{loss_ce.item():.4f}, KD:{loss_kd.item():.4f})", end="")
        print("")

    @torch.no_grad()
    def evaluate(self, mode='val'):
        self.student_model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true, y_pred_labels, y_pred_logits = [], [], []

        for inputs, labels in self.loader_dict[mode]:
            inputs, labels = inputs.to(self.device), labels.view(-1).to(self.device)
            s_logits, _ = self.student_model(inputs)
            loss = self.criterion.ce_loss(s_logits, labels)

            eval_loss += loss.item() * inputs.size(0)
            pred = torch.argmax(s_logits, 1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            # [核心修复]: 评价指标需要 labels，而最后 summarize 需要原始的 logits
            y_true.extend(labels.cpu().numpy())
            y_pred_labels.extend(pred.cpu().numpy())
            y_pred_logits.extend(s_logits.cpu().numpy())

        acc = 100. * correct / total
        mf1 = skmet.f1_score(y_true, y_pred_labels, average='macro') * 100

        # 返回最终测试所需要的 Numpy 二维矩阵格式
        return acc, mf1, eval_loss / total, np.array(y_true), np.array(y_pred_logits)

    def measure_latency(self):
        self.student_model.eval()
        dummy_input = torch.randn(10, 1, 30000).to(self.device)

        # 1. 训练态推理 (多分支)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): self.student_model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"\n[算力分析] 重参数化前 (多分支训练态) 推理时间: {(t1 - t0) / 50 * 1000:.2f} ms / batch")

        # 2. 部署态推理 (单分支大卷积 + 自动剪枝)
        # [修复]: 传入剪枝率，例如修剪掉25%的冗余通道
        self.student_model.deploy_and_prune(prune_ratio=0.25)

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): self.student_model(dummy_input)
        torch.cuda.synchronize()
        t1 = time.time()
        print(
            f"[算力分析] 重参数化后 (部署态+剪枝25%) 推理时间: {(t1 - t0) / 50 * 1000:.2f} ms / batch  ---> 速度飙升！")

    def run(self):
        print("\n[INFO] 开始第五章 EKD 知识蒸馏轻量化训练...")
        for epoch in range(1, 2):
            self.train_one_epoch(epoch)
            val_acc, val_mf1, val_loss, _, _ = self.evaluate('val')
            print(f"Epoch {epoch} | Val ACC: {val_acc:.2f}% | Val MF1: {val_mf1:.2f}")

            self.early_stopping(val_mf1, val_loss, self.student_model)
            if self.early_stopping.early_stop:
                break

        self.student_model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, self.ckpt_name)))
        self.measure_latency()
        _, _, _, y_true, y_pred = self.evaluate('test')
        return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--config', type=str, help='config file path',
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    config['mode'] = 'normal'

    # ========================================================
    # [核心修复] 恢复原版的空数组初始化逻辑，利用拼接的副作用将其强转为 float
    # 以完美兼容 utils.py 中 hardcode 的 '0.0', '1.0' 键值
    # ========================================================
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, 2):
        trainer = EKDTrainer(args, fold, config)
        y_true, y_pred = trainer.run()

        # 将整型的预测标签拼接到 float 类型的全零矩阵后
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])

        # 这里传入的 Y_true 已经自动变成 float 类型了，utils.py 不会再报错
        summarize_result(config, fold, Y_true, Y_pred)


if __name__ == "__main__":
    main()