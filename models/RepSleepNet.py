import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaborFourierPriorBank(nn.Module):
    def __init__(self, num_filters, kernel_size, sample_rate=100.0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        t = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size) / sample_rate
        self.register_buffer('t', t)
        self.f_gabor = torch.linspace(0.5, 20.0, num_filters // 2)
        self.f_fourier = torch.linspace(0.5, 40.0, num_filters // 2)

    def get_kernels(self):
        t = self.t.view(1, 1, -1)
        f_g = self.f_gabor.view(-1, 1, 1).to(t.device)
        gauss = torch.exp(-((t) ** 2) / (2 * 0.1 ** 2))
        gabor_kernels = gauss * torch.cos(2 * math.pi * f_g * t)

        f_f = self.f_fourier.view(-1, 1, 1).to(t.device)
        fourier_kernels = torch.cos(2 * math.pi * f_f * t)
        return torch.cat([gabor_kernels, fourier_kernels], dim=0)


class RepPhysConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.conv_branch = nn.Conv1d(in_channels, out_channels, kernel_size, stride, self.padding, bias=False)

        if in_channels == 1:
            self.phys_bank = GaborFourierPriorBank(out_channels, kernel_size)
            self.phys_scale = nn.Parameter(torch.ones(out_channels, 1, 1))

        self.deploy = False

    def forward(self, x):
        if self.deploy:
            return self.conv_branch(x)

        out_conv = self.conv_branch(x)
        if self.in_channels == 1:
            phys_weight = self.phys_bank.get_kernels() * self.phys_scale
            out_phys = F.conv1d(x, phys_weight, stride=self.stride, padding=self.padding)
            return out_conv + out_phys
        return out_conv

    def reparameterize(self):
        if self.deploy: return
        if self.in_channels == 1:
            phys_weight = self.phys_bank.get_kernels() * self.phys_scale
            self.conv_branch.weight.data = self.conv_branch.weight.data + phys_weight.data
        self.deploy = True
        if hasattr(self, 'phys_bank'):
            del self.phys_bank
            del self.phys_scale


class SToMe_TemporalBlock(nn.Module):
    def __init__(self, dim, threshold=0.85):
        super().__init__()
        self.local_tcn = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.threshold = threshold

    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_t = F.gelu(self.local_tcn(x_t)) + x_t
        x_out = x_t.transpose(1, 2)

        if self.training:
            return x_out

        sim = F.cosine_similarity(x_out[:, :-1, :], x_out[:, 1:, :], dim=-1)
        mask = (sim > self.threshold).float().unsqueeze(-1)
        smoothed_x = x_out.clone()
        smoothed_x[:, 1:, :] = x_out[:, 1:, :] * (1 - mask) + ((x_out[:, :-1, :] + x_out[:, 1:, :]) / 2) * mask
        return smoothed_x


class RepSleepNet(nn.Module):
    def __init__(self, num_classes=5, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = 128

        self.spatial_stem = nn.Sequential(
            RepPhysConv1d(1, 64, kernel_size=63, stride=4),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),
            RepPhysConv1d(64, self.feature_dim, kernel_size=31, stride=2),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.stome_layer = SToMe_TemporalBlock(dim=self.feature_dim)
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self.register_buffer('channel_mask', torch.ones(self.feature_dim))

    def forward(self, x):
        B = x.shape[0]
        # 自动计算每个Epoch的长度，避免硬编码 3000
        epoch_len = x.shape[-1] // self.seq_len

        # 1. 拆分为[B, seq_len, epoch_len]
        x = x.view(B, self.seq_len, epoch_len)
        # 2. 融合成[B*seq_len, 1, epoch_len]
        x = x.view(B * self.seq_len, 1, epoch_len)

        feat = self.spatial_stem(x)
        feat = feat.squeeze(-1)

        feat = feat * self.channel_mask.view(1, -1)

        feat_seq = feat.view(B, self.seq_len, self.feature_dim)
        feat_seq = self.stome_layer(feat_seq)

        # ========================================
        # [核心修复]: 聚合上下文序列帧 -> 获得整体特征
        # ========================================
        feat_pooled = feat_seq.mean(dim=1)  # 形状变为: [B, 128]
        logits = self.fc(feat_pooled)  # 形状变为: [B, 5]

        # 返回 logits 用于交叉熵，返回 feat_pooled 用于特征蒸馏
        return logits, feat_pooled

    def deploy_and_prune(self, prune_ratio=0.2):
        """
        部署与剪枝一体化核心函数：
        1. 首先执行结构重参数化。
        2. 计算BN层权重的L1范数，利用分位数执行物理通道剪枝。
        """
        # 1. 重参数化
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()

        # 2. 物理剪枝 (基于解释性打分/BN权重相对大小)
        bn_layer = self.spatial_stem[5]
        gamma = bn_layer.weight.data.abs()

        # 计算动态分位数阈值，强行剪掉最不重要的前 prune_ratio (如20%)
        threshold = torch.quantile(gamma, prune_ratio)

        # 寻找存活的通道索引
        alive_indices = torch.nonzero(gamma > threshold).squeeze()
        print(
            f"\n[INFO] 剪枝完成：保留了 {len(alive_indices)}/{self.feature_dim} 个通道！ (剪枝率: {prune_ratio * 100}%)")

        # 将低于阈值的通道特征拦截置零
        self.channel_mask[gamma <= threshold] = 0.0



'''
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

model = RepSleepNet().cuda()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数量 (Total Trainable Params): {total_params} M")

x = torch.rand([2, 1, 30000]).cuda()
out = model(x)
print(out, len(out), out[0].shape, out[1].shape)
'''