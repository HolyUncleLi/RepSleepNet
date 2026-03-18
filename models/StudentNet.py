# --- models/ekd_student.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# ==========================================
# 1. 基础轻量化算子 (源自第三章)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, max(1, in_dim // reduction), bias=False),
            nn.ReLU(),
            nn.Linear(max(1, in_dim // reduction), in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x).unsqueeze(-1)
        return x * weights


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel):
        super().__init__()
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.lkb_origin = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        if small_kernel is not None:
            self.small_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, small_kernel, stride, small_kernel // 2, groups=groups,
                          bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(x)
        return out


class LKConv_Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ReparamLargeKernelConv(in_channels, out_channels, kernel_size=31, stride=1, groups=1, small_kernel=5))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(SEBlock(out_channels))
            layers.append(nn.GELU())
            in_channels = out_channels
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)


def FFT_for_Period(x, k=3):
    xf = torch.fft.rfft(x.view(x.shape[0], -1), dim=1)
    frequency_list = abs(xf)
    signal_len = x.view(x.shape[0], -1).shape[1]
    top_list = torch.topk(frequency_list, k, dim=1)[1]
    top_list[top_list == 0] = 1
    period = signal_len // top_list
    weight = torch.topk(frequency_list, k, dim=1)[0]
    return period.mean(0).to(torch.int), weight


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super().__init__()
        kernels = []
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x):
        res_list = [k(x) for k in self.kernels]
        return torch.stack(res_list, dim=-1).mean(-1)


class TimesBlock(nn.Module):
    def __init__(self, d_model, seq_len, top_k=3):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V2(d_model, d_model * 2, num_kernels=3),
            nn.GELU(),
            Inception_Block_V2(d_model * 2, d_model, num_kernels=3),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i].item()
            if period > self.seq_len: period = self.seq_len
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - self.seq_len), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


# ==============================================================================
# 2. 第五章核心模型：EKD 学生网络
# ==============================================================================
class StudentNet(nn.Module):
    def __init__(self, config):
        super(StudentNet, self).__init__()
        self.cfg = config
        teacher_feat_dim = self.cfg['classifier'].get('afr_reduced_dim', 128)
        num_prototypes = self.cfg['classifier']['prototype_num']
        num_classes = self.cfg['classifier']['num_classes']

        # 1. 骨干特征提取 (建身体)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4, 4)
        )
        self.lk_backbone = LKConv_Stage(in_channels=32, out_channels=64, num_blocks=2)
        # 对齐教师的特征维度
        self.feature_align = nn.Conv1d(64, teacher_feat_dim, kernel_size=1)

        # 2. 时序折叠捕获长期依赖 (建大脑)
        # 根据 config 中的输入长度预估时序序列长度
        # 如果输入 3000，经过 stem 后长度大约是 187
        self.seq_len = 187
        self.lk_times = TimesBlock(d_model=teacher_feat_dim, seq_len=self.seq_len, top_k=3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 模拟教师交叉注意力的距离估算
        self.proto_distance_estimator = nn.Linear(teacher_feat_dim, num_prototypes)

        # 3. 分类器
        self.bn = nn.BatchNorm1d(num_prototypes)
        self.fc = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        # 1. 特征提取与对齐
        x = self.stem(x)
        x = self.lk_backbone(x)
        aligned_features = self.feature_align(x)  # [B, 128, L]

        # 2. 时序折叠处理
        t_feat = aligned_features.permute(0, 2, 1)  # [B, L, C]
        t_feat = self.lk_times(t_feat)
        t_feat = t_feat.permute(0, 2, 1)  # [B, C, L]

        # 3. 估算 M 个原型的距离
        pooled_feat = self.global_pool(t_feat).squeeze(-1)  # [B, 128]
        est_distances = self.proto_distance_estimator(pooled_feat)  # [B, M]

        # 4. 模拟原型匹配打分与分类
        similarity = torch.log((est_distances + 1) / (est_distances + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        # 学生网络必须同时返回这三个张量，以支持解释性知识蒸馏
        return logits, aligned_features, est_distances


import math
import warnings
import argparse
import os
import json
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', type=int, default=49, help='random seed')
parser.add_argument('--gpu', type=str, default="0", help='gpu id')
parser.add_argument('--config', type=str, help='config file path',
                    default='./SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

with open(args.config) as config_file:
    config = json.load(config_file)
config['name'] = os.path.basename(args.config).replace('.json', '')
config['mode'] = 'normal'

model = StudentNet(config).cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量 (Total Trainable Params): {total_params} M")

x = torch.rand([64, 1, 30000]).cuda()
out = model(x)
print(out, out.shape)
