import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.L = dim
        self.D = 128
        self.K = 1
        
        self.attn_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attn_w = nn.Linear(self.D, self.K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        A = self.attn_w(self.attn_V(x) * self.attn_U(x))
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        return A


class CLAM_SB(nn.Module):
    def __init__(self, input_dim: int = 512, n_classes: int = 6, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn_net = GatedAttention(512)
        self.classifiers = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        A = self.attn_net(h)
        M = torch.mm(A, h)
        logits = self.classifiers(M)
        return logits, A
