import torch
import torch.nn as nn
import numpy as np

def morton_code(x, y, z, bits=16):
    def part1by1(n):
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    x = x.long()
    y = y.long()
    z = z.long()
    max_val = 2**bits - 1
    x = (x * max_val).clamp(0, max_val)
    y = (y * max_val).clamp(0, max_val)
    z = (z * max_val).clamp(0, max_val)
    return (part1by1(x) << 2) | (part1by1(y) << 1) | part1by1(z)

def z_order_sort(points):
    min_vals = points.min(dim=0, keepdim=True)[0]
    max_vals = points.max(dim=0, keepdim=True)[0]
    norm_pts = (points - min_vals) / (max_vals - min_vals + 1e-8)
    codes = morton_code(norm_pts[:,0], norm_pts[:,1], norm_pts[:,2])
    return torch.argsort(codes)


class DownsampleNet(nn.Module):
    """
    下采样网络：通过最远点采样（FPS）得到关键点，并用 PointNet 风格的 MLP 提取特征。
    """
    def __init__(self, in_channels=3, out_channels=128, num_keypoints=512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )

    def farthest_point_sample(self, points, npoint):
        """
        points: (B, N, 3)
        """
        # 仅支持单 batch
        assert points.dim() == 2
        N = points.shape[0]
        if N <= npoint:
            return torch.arange(N, device=points.device)
        # 此处省略实现，直接返回前 npoint 个点（实际应使用 FPS）
        # 注意：实际应用中可调用 pytorch3d 或自己实现 FPS
        # 为简化，这里随机采样
        idx = torch.randperm(N)[:npoint]
        return idx

    def forward(self, points):
        """
        points: (N, 3)
        返回: key_points (num_keypoints, 3), features (num_keypoints, out_channels)
        """
        N = points.shape[0]
        # 1. 最远点采样得到关键点索引
        if N > self.num_keypoints:
            fps_idx = self.farthest_point_sample(points, self.num_keypoints)
        else:
            fps_idx = torch.arange(N, device=points.device)
        key_pts = points[fps_idx]

        # 2. 提取特征（对所有原始点提取特征，然后插值到关键点，或直接在关键点上 MLP）
        # 为简化，我们在关键点上应用 MLP（但这样会丢失局部上下文）
        # 更好的做法：使用 PointNet++ 的 set abstraction，但这里简单用 MLP 在关键点上
        features = self.mlp(key_pts)  # (num_keypoints, out_channels)
        return key_pts, features

# ------------------- 对称变换预测网络（STPNet）-------------------
class STPNet(nn.Module):
    """
    预测每个点的仿射变换矩阵和平移向量。
    输入: features (N_k, D)
    输出: A (N_k, 3, 3), T (N_k, 3)
    """
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        # 预测 3x3 仿射矩阵的 9 个参数 + 3 个平移参数
        self.mlp_A = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9)   # 3x3 矩阵
        )
        self.mlp_T = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, features):
        A_flat = self.mlp_A(features)   # (N_k, 9)
        T = self.mlp_T(features)        # (N_k, 3)
        A = A_flat.view(-1, 3, 3)       # (N_k, 3, 3)
        return A, T

class SIO(nn.Module):
    """
    Symmetry-aware Interleaved Ordering Network。
    Input: partial point cloud P (N, 3)
    Output: causal sequence P_causal (2*N_k, 3)
    """
    def __init__(self,
                 num_keypoints=512,
                 feature_dim=128,
                 serialization='zorder'):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.feature_dim = feature_dim
        self.serialization = serialization

        self.downsample = DownsampleNet(in_channels=3,
                                        out_channels=feature_dim,
                                        num_keypoints=num_keypoints)
        self.stpnet = STPNet(in_channels=feature_dim, hidden_dim=128)

    def forward(self, points):
        """
        points: (N, 3)
        """
        # 1. downsample
        P_k, F_k = self.downsample(points)       # (N_k, 3), (N_k, D)

        # 2. ordering
        sorted_idx = z_order_sort(P_k)
        P_ordered = P_k[sorted_idx]              # (N_k, 3)

        # 3. estimates per-point affine matrix
        A, T = self.stpnet(F_k)              # (N_k, 3, 3), (N_k, 3)
        F_k_sorted = F_k[sorted_idx]
        A_sorted = A[sorted_idx]
        T_sorted = T[sorted_idx]
        P_sym = torch.bmm(P_ordered.unsqueeze(1), A_sorted.transpose(1,2)).squeeze(1) + T_sorted

        # 4. interleaving
        N_k = P_ordered.shape[0]
        P_causal = torch.stack([P_ordered, P_sym], dim=1).view(-1, 3)   # (2*N_k, 3)

        return P_causal
