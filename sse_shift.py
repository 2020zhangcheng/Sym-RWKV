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
            F (torch.Tensor): input features of shape (B, N, d_model)
        Returns:
            tuple: (F_R, F_K, F_V) each of shape (B, N, d_model)
        """
        B, N, C = F.shape
        # Pad sequence along the N dimension (index 1)
        # pad tuple: (left, right, top, bottom) = (0, 0, 2, 2) means pad N by 2 on both sides
        F_pad = torch.nn.functional.pad(F, (0, 0, 2, 2), mode='replicate')  # (B, N+4, C)

        # Position indices in the original sequence (0-based), shifted by 2 due to padding
        pos = torch.arange(N, device=F.device) + 2  # (N,)

        # Extract slices
        F1 = F_pad[:, pos - 4, :self.d_slice]                 # (B, N, d_slice)
        F2 = F_pad[:, pos - 2, self.d_slice:2*self.d_slice]   # (B, N, d_slice)
        F3 = F_pad[:, pos,     2*self.d_slice:3*self.d_slice] # (B, N, d_slice)
        F4 = F_pad[:, pos + 2, 3*self.d_slice:]               # (B, N, d_slice)

        # Concatenate along feature dimension to form F'
        F_prime = torch.cat([F1, F2, F3, F4], dim=2)         # (B, N, d_model)

        # Expand gating vectors to (1,1,d_model) for broadcasting
        mu_R = self.mu_R.view(1, 1, -1)
        mu_K = self.mu_K.view(1, 1, -1)
        mu_V = self.mu_V.view(1, 1, -1)

        # Apply gating
        out_R = F + (1 - mu_R) * F_prime
        out_K = F + (1 - mu_K) * F_prime
        out_V = F + (1 - mu_V) * F_prime

        return out_R, out_K, out_V
