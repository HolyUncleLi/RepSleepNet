#--- 划分不同剪枝率

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
import sklearn.metrics as skmet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 导入你的自定义模块
from utils import set_random_seed
from loader import EEGDataLoader
from models.RepSleepNet import RepSleepNet

import warnings

warnings.filterwarnings("ignore")


def count_model_params(model):
    """统计模型参数量 (M)"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1e6


@torch.no_grad()
def evaluate_metrics(model, test_loader, device):
    """评估模型的 Acc 和 MF1"""
    model.eval()
    correct, total = 0, 0
    y_true, y_pred_labels = [], []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.view(-1).to(device)
        logits, _ = model(inputs)

        predicted = torch.argmax(logits, 1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().numpy())
        y_pred_labels.extend(predicted.cpu().numpy())

    acc = 100. * correct / total
    mf1 = skmet.f1_score(y_true, y_pred_labels, average='macro') * 100
    return acc, mf1


@torch.no_grad()
def measure_latency(model, device):
    """测量模型推理耗时 (ms/batch)"""
    model.eval()
    dummy_input = torch.randn(10, 1, 30000).to(device)  # 模拟 BatchSize=10

    # 预热 GPU
    for _ in range(10):
        model(dummy_input)
    torch.cuda.synchronize()

    # 正式测量
    t0 = time.time()
    for _ in range(50):
        model(dummy_input)
    torch.cuda.synchronize()
    t1 = time.time()

    latency = (t1 - t0) / 50 * 1000  # ms/batch
    return latency


def run_evaluation(args, cfg, device, csv_save_path):
    """
    模块一：运行模型测试，并将结果保存为本地 CSV 文件
    """
    fold = 1
    print(f"[INFO] 初始化第一折 (Fold {fold}) 数据集...")
    test_dataset = EEGDataLoader(cfg, fold, set='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['training_params']['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)

    ckpt_path = os.path.join('checkpoints', 'RepSleepNet_' + str(args.seed), f'repsleep_fold-{fold:02d}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] 找不到模型权重: {ckpt_path}")

    results = []

    print(f"\n{'=' * 50}")
    print(f" 开始执行剪枝率消融实验 (Fold: {fold})")
    print(f"{'=' * 50}")

    for ratio in args.prune_steps:
        # 重新初始化模型并加载原始权重
        model = RepSleepNet(num_classes=cfg['classifier']['num_classes'],
                            seq_len=cfg['dataset'].get('seq_len', 10)).to(device)

        state_dict = torch.load(ckpt_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)

        # 执行部署重参数化与剪枝
        model.deploy_and_prune(prune_ratio=ratio)

        # 测试各项指标
        acc, mf1 = evaluate_metrics(model, test_loader, device)
        lat = measure_latency(model, device)
        base_params = count_model_params(model)
        theoretical_params = base_params * (1 - ratio)

        print(
            f"[Ratio {ratio:.1f}] Acc: {acc:.2f}% | MF1: {mf1:.2f} | Latency: {lat:.2f}ms | Params: {theoretical_params:.3f}M")

        results.append({
            'Prune_Ratio': ratio,
            'Accuracy': acc,
            'MF1': mf1,
            'Latency_ms': lat,
            'Params_M': theoretical_params
        })

    # 保存到本地 CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_save_path, index=False)
    print(f"\n[SUCCESS] 测试完成！测试数据已安全保存至: {csv_save_path}")


def plot_results(csv_path, img_save_path):
    """
    模块二：读取本地 CSV 数据文件，并绘制折线图
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] 找不到数据文件 {csv_path}。请先将 DO_EVALUATE 设为 True 跑一遍数据。")

    print(f"[INFO] 正在读取本地数据文件: {csv_path}，开始绘制折线图...")
    df = pd.read_csv(csv_path)

    ratios = df['Prune_Ratio'].values
    acc = df['Accuracy'].values
    mf1 = df['MF1'].values
    latency = df['Latency_ms'].values
    params = df['Params_M'].values

    # 设置画板大小，包含3个子图 (1行3列)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('RepSleepNet Pruning Ablation Study', fontsize=16, fontweight='bold')

    # 图1：准确率与 MF1 分数
    ax1 = axes[0]
    ax1.plot(ratios, acc, marker='o', color='b', label='Accuracy (%)')
    ax1.plot(ratios, mf1, marker='s', color='g', label='Macro F1')
    ax1.set_xlabel('Pruning Ratio', fontsize=12)
    ax1.set_ylabel('Performance Metric', fontsize=12)
    ax1.set_title('Accuracy and MF1', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 图2：推理耗时 Latency
    ax2 = axes[1]
    ax2.plot(ratios, latency, marker='^', color='r', linewidth=2)
    ax2.set_xlabel('Pruning Ratio', fontsize=12)
    ax2.set_ylabel('Latency', fontsize=12)
    ax2.set_title('Inference Latency', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 图3：模型参数量 Params
    ax3 = axes[2]
    ax3.plot(ratios, params, marker='D', color='purple', linewidth=2)
    ax3.set_xlabel('Pruning Ratio', fontsize=12)
    ax3.set_ylabel('Parameters(M)', fontsize=12)
    ax3.set_title('Model Parameters', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.6)

    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(img_save_path, dpi=300)
    print(f"[SUCCESS] 折线图已成功生成并保存至: {img_save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--config', type=str,
                        default='./configs/SleePyCo-Transformer_SL-10_numScales-3_Sleep-EDF-2013_wavesensing.json')
    parser.add_argument('--prune_steps', nargs='+', type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed, use_cuda=torch.cuda.is_available())

    with open(args.config) as config_file:
        cfg = json.load(config_file)
    cfg['mode'] = 'normal'

    os.makedirs("results", exist_ok=True)

    # 统一的数据与图片存放路径
    csv_save_path = "results/pruning_ablation_data_fold1.csv"
    img_save_path = "results/pruning_ablation_plot_fold1.png"

    # =========================================================================
    # 控制台：你可以通过修改下方的两个 True/False 来控制脚本的行为
    # =========================================================================
    DO_EVALUATE = False  # 【开关】是否运行模型测试，并将结果保存到本地 CSV
    DO_PLOT = True  # 【开关】是否读取本地的 CSV 数据，绘制并保存折线图图片
    # =========================================================================

    if DO_EVALUATE:
        run_evaluation(args, cfg, device, csv_save_path)

    if DO_PLOT:
        plot_results(csv_save_path, img_save_path)


if __name__ == "__main__":
    main()