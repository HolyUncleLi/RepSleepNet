# ==============================================================================
# 第五章：基于解释性知识蒸馏(EKD)的轻量化睡眠分期网络全解
# 包含：教师模型(Teacher)、轻量化算子库、学生模型(Student)
# ==============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


# ==============================================================================
# 第一部分：第三章轻量化高效算子库 (用于构建学生模型)
# ==============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""

    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_dim // reduction, in_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.layers(x).unsqueeze(-1)
        return x * weights


class ReparamLargeKernelConv(nn.Module):
    """大核卷积模块 (重参数化支持)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
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
    """学生网络的主干特征提取器 (替代教师的双流Gabor)"""

    def __init__(self, in_channels, out_channels, num_blocks=2):
        super().__init__()
        layers = []
        dmodel = out_channels
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
    """提取周期特征 (LKTimes 核心)"""
    xf = torch.fft.rfft(x.view(x.shape[0], -1), dim=1)
    frequency_list = abs(xf)
    signal_len = x.view(x.shape[0], -1).shape[1]

    # 简化取Top-K频率
    top_list = torch.topk(frequency_list, k, dim=1)[1]
    top_list[top_list == 0] = 1  # 避免除以0
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
    """时序折叠模块 (LKTimes，替代交叉注意力)"""

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

            # 1D to 2D Folding
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


# ==============================================================================
# 第二部分：第五章核心 - EKD 学生模型 (Student)
# ==============================================================================

class EKD_StudentNet(nn.Module):
    """
    轻量化解释性知识蒸馏学生网络：
    1. 用 LKConv 替代 Gabor 双流 (提效)
    2. 用 LKTimes 替代 Cross-Attention 原型匹配 (提效)
    3. 保留特征对齐接口与距离输出接口，用于接收教师指导 (传授灵魂)
    """

    def __init__(self, in_channels=1, teacher_feat_dim=128, num_prototypes=60, num_classes=5):
        super(EKD_StudentNet, self).__init__()

        # 1. 骨干：面向原型语义逼近的轻量化特征提取模块
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4, 4)
        )
        self.lk_backbone = LKConv_Stage(in_channels=32, out_channels=64, num_blocks=2)

        # [蒸馏接口1] 特征维度对齐：映射到教师特征维度 (如128)
        self.feature_align = nn.Conv1d(64, teacher_feat_dim, kernel_size=1)

        # 2. 桥梁：融合特征折叠机制的高效时序模块 (替换原有的多子空间交叉匹配 SCM)
        # 注意：这里的 seq_len 需要与前面 stem 和 conv 降采样后的序列长度匹配
        # 假设输入 3000，经过 /4 和 /4 后大约是 187
        self.lk_times = TimesBlock(d_model=teacher_feat_dim, seq_len=187, top_k=3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # [蒸馏接口2] 模拟原型交叉匹配的逻辑距离评估 (直接生成M个标量距离)
        self.proto_distance_estimator = nn.Linear(teacher_feat_dim, num_prototypes)

        # 3. 分类头 (与教师模型保持一致的计算逻辑)
        self.bn = nn.BatchNorm1d(num_prototypes)
        self.fc = nn.Linear(num_prototypes, num_classes)

    def forward(self, x):
        # x:[B, 1, 3000]

        # 1. 特征提取与维度对齐
        x = self.stem(x)
        x = self.lk_backbone(x)
        aligned_features = self.feature_align(x)  # [B, 128, 187]

        # 2. 时序折叠捕获长期依赖
        # TimesBlock 输入需为 [B, L, C]
        t_feat = aligned_features.permute(0, 2, 1)
        t_feat = self.lk_times(t_feat)
        t_feat = t_feat.permute(0, 2, 1)  # 返回 [B, C, L]

        # 3. 生成对应 M 个原型的距离估算值
        pooled_feat = self.global_pool(t_feat).squeeze(-1)  # [B, 128]
        est_distances = self.proto_distance_estimator(pooled_feat)  # [B, M]

        # 4. 模拟原型匹配的决策输出
        similarity = torch.log((est_distances + 1) / (est_distances + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        # 返回分类结果，以及用于蒸馏对齐的中间层特征和伪距离向量
        return logits, aligned_features, est_distances


# ==============================================================================
# 第三部分：第四章重型可解释模型 (Teacher，极简脱水版，仅保留网络结构)
# ==============================================================================
# 为了保证单个文件可运行，保留了核心的LG-WDS和SCM模块

class LearnableGaborConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sample_rate=100.0):
        super().__init__()
        self.out_channels, self.padding = out_channels, kernel_size // 2
        self.mu_f = nn.Parameter(torch.rand(out_channels) * 30.0 + 0.5)
        self.sigma = nn.Parameter(torch.ones(out_channels) * 10.0)
        t = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size) / sample_rate
        self.register_buffer('t', t)

    def forward(self, x):
        t = self.t.view(1, 1, -1)
        envelope = torch.exp(-0.5 * (t ** 2) / (self.sigma.view(-1, 1, 1) ** 2))
        carrier_cos = torch.cos(2 * math.pi * self.mu_f.view(-1, 1, 1) * t)
        carrier_sin = torch.sin(2 * math.pi * self.mu_f.view(-1, 1, 1) * t)
        out_real = F.conv1d(x, envelope * carrier_cos, padding=self.padding)
        out_imag = F.conv1d(x, envelope * carrier_sin, padding=self.padding)
        magnitude = torch.sqrt(out_real ** 2 + out_imag ** 2 + 1e-8)
        return magnitude, out_real


class LGWDS_Net(nn.Module):
    """教师网络的特征提取器 (双流重型结构)"""

    def __init__(self, out_dim=128):
        super().__init__()
        self.gabor_layer = LearnableGaborConv1d(1, 64, kernel_size=63)
        self.semantic_stream = nn.Sequential(nn.Conv1d(64, 128, kernel_size=31, stride=16), nn.GELU())  # 简化的语义流
        self.morph_stream = nn.Sequential(nn.Conv1d(64, 128, kernel_size=21, padding=10), nn.GELU())  # 简化的形态流
        self.project = nn.Conv1d(128, 128, 1)
        self.gate = nn.Sequential(nn.Conv1d(256, 1, 1), nn.Sigmoid())
        self.final_proj = nn.Conv1d(128, out_dim, 1)
        self.final_pool = nn.AdaptiveAvgPool1d(187)  # 对齐学生维度

    def forward(self, x):
        mag, raw_real = self.gabor_layer(x)
        sem_feat = self.semantic_stream(mag)
        morph_feat = self.morph_stream(raw_real)
        # 全息融合
        sem_up = F.interpolate(sem_feat, size=morph_feat.shape[-1], mode='linear')
        mask = self.gate(torch.cat([morph_feat, self.project(sem_up)], dim=1))
        fused = morph_feat * (1 + mask * self.project(sem_up))
        out = self.final_pool(self.final_proj(fused))
        return out


class Teacher_ProtoPNet(nn.Module):
    """
    第四章教师网络
    蒸馏关键：加入 return_all_for_kd 控制符，返回用于指导学生的隐层状态
    """

    def __init__(self, num_classes=5, num_prototypes=60, feat_dim=128):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.feature_extractor = LGWDS_Net(out_dim=feat_dim)

        # 极简版的交叉匹配 (SCM)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim, 1))

        self.bn = nn.BatchNorm1d(num_prototypes)
        self.fc = nn.Linear(num_prototypes, num_classes)

    def forward(self, x, return_all_for_kd=False):
        # 1. 教师特征提取
        features = self.feature_extractor(x)  # [B, 128, 187]

        # 2. 原型距离计算 (简化的欧氏距离/相似度计算)
        B, C, L = features.shape
        f_flat = features.view(B, C, L).unsqueeze(1)  # [B, 1, C, L]
        p_flat = self.prototypes.unsqueeze(0).expand(B, -1, -1, -1)  # [B, M, C, 1]

        # 计算特征与原型的距离 (这里用均方差模拟 SCM 的匹配)
        distances = torch.mean((f_flat - p_flat) ** 2, dim=[2, 3])  # [B, M]

        similarity = torch.log((distances + 1) / (distances + 1e-4))
        bn_similarity = self.bn(similarity)
        logits = self.fc(bn_similarity)

        if return_all_for_kd:
            # 蒸馏所需：最终分类、LG-WDS特征图、SCM距离输出
            return logits, features, distances
        return logits


# ==============================================================================
# 第四部分：解释性蒸馏流程演示 (如何联合使用以上模型)
# ==============================================================================

def EKD_Loss_Function(s_logits, s_feat, s_dist, t_logits, t_feat, t_dist, labels):
    """
    第五章核心：解释性知识蒸馏联合损失函数
    """
    alpha, beta, temperature = 10.0, 5.0, 2.0

    # 1. 基础分类损失 (让学生学会做题)
    loss_ce = F.cross_entropy(s_logits, labels)

    # 2. 特征级对齐损失 (让学生的 LKConv 学会像 Gabor 一样提取波形特征)
    loss_feat = F.mse_loss(s_feat, t_feat.detach())

    # 3. 逻辑级对齐损失 (让学生的 LKTimes 学会像 SCM交叉注意力一样给原型打分)
    t_prob = F.softmax(-t_dist.detach() / temperature, dim=1)
    s_log_prob = F.log_softmax(-s_dist / temperature, dim=1)
    loss_kd = F.kl_div(s_log_prob, t_prob, reduction='batchmean')

    total_loss = loss_ce + alpha * loss_feat + beta * loss_kd
    return total_loss


if __name__ == "__main__":
    # 测试代码流转
    batch_size = 4
    seq_len = 3000  # 30s EEG 信号 (100Hz)

    dummy_input = torch.randn(batch_size, 1, seq_len)
    dummy_labels = torch.randint(0, 5, (batch_size,))

    # 初始化教师并冻结
    teacher = Teacher_ProtoPNet().eval()
    for p in teacher.parameters(): p.requires_grad = False

    # 初始化学生
    student = EKD_StudentNet()

    # 前向传播 (模拟蒸馏过程)
    with torch.no_grad():
        t_logits, t_feat, t_dist = teacher(dummy_input, return_all_for_kd=True)

    s_logits, s_feat, s_dist = student(dummy_input)

    # 计算 EKD 损失
    loss = EKD_Loss_Function(s_logits, s_feat, s_dist, t_logits, t_feat, t_dist, dummy_labels)

    print(f"Teacher Feature Shape: {t_feat.shape}")
    print(f"Student Feature Shape: {s_feat.shape}  --> Perfect Match!")
    print(f"Teacher Distances Shape: {t_dist.shape}")
    print(f"Student Distances Shape: {s_dist.shape}  --> Perfect Match!")
    print(f"Total EKD Loss: {loss.item():.4f}")
    print("知识蒸馏闭环成功运行！")