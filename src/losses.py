import torch
import torch.nn.functional as F


def calc_kd_loss(logits_s: torch.Tensor, logits_t: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    log_p = F.log_softmax(logits_s / T, dim=1)
    q = F.softmax(logits_t / T, dim=1)
    return F.kl_div(log_p, q, reduction='batchmean') * (T ** 2)


def calc_akd_loss(attn_s: torch.Tensor, attn_t: torch.Tensor) -> torch.Tensor:
    log_s = torch.log(attn_s + 1e-8)
    return F.kl_div(log_s, attn_t, reduction='batchmean')
