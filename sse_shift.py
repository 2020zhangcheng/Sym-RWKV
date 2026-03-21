import torch
import torch.nn as nn

class SSEShift(nn.Module):
    """
    Symmetry-neighbor State Expansion (SSE) Shift module.
    For each token at position n, constructs F' by concatenating:
        F1 = F[n-2, 0:D/4]
        F2 = F[n-1, D/4:D/2]
        F3 = F[n+1, D/2:3D/4]
        F4 = F[n+2, 3D/4:D]
    Then computes SSE-Shift_{(R/K/V)} = F + (1 - μ_{(R/K/V)}) * F'
    where μ are learnable gating vectors.

    Args:
        d_model (int): feature dimension (must be divisible by 4).
    """
    def __init__(self, d_model):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        self.d_model = d_model
        self.d_slice = d_model // 4

        # Learnable gating vectors for R, K, V (initialized to 0)
        self.mu_R = nn.Parameter(torch.zeros(d_model))
        self.mu_K = nn.Parameter(torch.zeros(d_model))
        self.mu_V = nn.Parameter(torch.zeros(d_model))

    def forward(self, F):
        """
        Args:
            F (torch.Tensor): input features of shape (seq_len, d_model)
        Returns:
            tuple: (F_R, F_K, F_V) each of shape (seq_len, d_model)
        """
        seq_len, d = F.shape

        # Pad sequence to handle boundaries (replicate edge values)
        # Pad left by 2 and right by 2 for indices -2, -1, +1, +2
        F_pad = torch.nn.functional.pad(F, (0, 0, 2, 2), mode='replicate')  # (seq_len+4, d)

        # Original indices in padded tensor: shift by +2
        idx = torch.arange(seq_len, device=F.device) + 2

        # Extract slices
        F1 = F_pad[idx - 4, :self.d_slice]                 # (seq_len, d_slice)
        F2 = F_pad[idx - 2, self.d_slice:2*self.d_slice]   # (seq_len, d_slice)
        F3 = F_pad[idx,     2*self.d_slice:3*self.d_slice] # (seq_len, d_slice)
        F4 = F_pad[idx + 2, 3*self.d_slice:]               # (seq_len, d_slice)

        # Concatenate to form F'
        F_prime = torch.cat([F1, F2, F3, F4], dim=1)       # (seq_len, d_model)

        # Apply gating
        out_R = F + (1 - self.mu_R) * F_prime
        out_K = F + (1 - self.mu_K) * F_prime
        out_V = F + (1 - self.mu_V) * F_prime

        return out_R, out_K, out_V
