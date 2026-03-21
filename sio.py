import torch
import torch.nn as nn
import numpy as np
from .model_utils import MLP_CONV, Transformer, PointNet_SA_Module_KNN

# ---------- Helper: Z-order (Morton code) sorting ----------
def morton_code(x, y, z, bits=16):
    """Compute Morton code (Z-order curve) for 3D points."""
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
    """
    Sort point cloud using Z-order (Morton code).
    points: (N, 3)
    """
    # Normalize to [0,1]
    min_vals = points.min(dim=0, keepdim=True)[0]
    max_vals = points.max(dim=0, keepdim=True)[0]
    norm_pts = (points - min_vals) / (max_vals - min_vals + 1e-8)
    codes = morton_code(norm_pts[:,0], norm_pts[:,1], norm_pts[:,2])
    return torch.argsort(codes)

# ---------- SIO Module ----------
class SIO(nn.Module):
    """
    Symmetry-aware Interleaved Ordering (SIO) module.
    Input: point cloud (B, N, 3)
    Output: causal sequence (B, 1024, 3)
    """
    def __init__(self, out_dim=512, serialization='zorder'):
        super(SIO, self).__init__()
        self.serialization = serialization
        # Original LSTNet components (unchanged)
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128, 256], group_all=False, if_bn=False, if_idx=True)
        self.mlp = nn.Sequential(
            nn.Linear(256*2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 9+3)
        )

    def forward(self, point_cloud):
        """
        point_cloud: (B, N, 3)
        """
        B, N, _ = point_cloud.shape
        # Transpose to (B, 3, N) to match the original module's expectation
        xyz = point_cloud.transpose(1, 2).contiguous()  # (B, 3, N)

        # 1. Downsample to obtain keypoints and features (same as LSTNet)
        keypoints, keyfeatures, _ = self.sa_module_1(xyz, xyz)   # keypoints: (B,3,512), keyfeatures: (B,256,512)
        feat = keyfeatures.transpose(2, 1).contiguous()            # (B,512,256)

        # Global feature
        gf_feat = feat.max(dim=1, keepdim=True)[0]                # (B,1,256)
        feat = torch.cat([feat, gf_feat.repeat(1, feat.size(1), 1)], dim=-1)  # (B,512,512)

        # Predict affine matrices and translation vectors
        ret = self.mlp(feat)                                     # (B,512,12)
        R = ret[:, :, :9].view(B, 512, 3, 3)                     # (B,512,3,3)
        T = ret[:, :, 9:]                                        # (B,512,3)

        # Compute symmetric points
        keypoints_t = keypoints.transpose(2, 1).contiguous()      # (B,512,3)
        sym_points = torch.matmul(keypoints_t.unsqueeze(2), R).view(B, 512, 3) + T  # (B,512,3)

        # 2. Sort keypoints (per batch) using the selected ordering strategy
        sorted_indices = []
        for i in range(B):
            pts = keypoints_t[i]          # (512,3)
            if self.serialization == 'zorder':
                idx = z_order_sort(pts)
            else:
                raise ValueError(f"Unsupported serialization: {self.serialization}")
            sorted_indices.append(idx)
        sorted_indices = torch.stack(sorted_indices, dim=0)       # (B,512)

        # Reorder keypoints and symmetric points accordingly
        sorted_keypoints = torch.gather(keypoints_t, 1,
                                        sorted_indices.unsqueeze(-1).expand(-1, -1, 3))  # (B,512,3)
        sorted_sym = torch.gather(sym_points, 1,
                                  sorted_indices.unsqueeze(-1).expand(-1, -1, 3))       # (B,512,3)

        # 3. Interleave to form a causal sequence (B, 1024, 3)
        interleaved = torch.stack([sorted_keypoints, sorted_sym], dim=2)  # (B,512,2,3)
        causal_seq = interleaved.view(B, -1, 3)                           # (B,1024,3)

        return causal_seq
