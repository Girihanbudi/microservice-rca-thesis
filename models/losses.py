import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    depth = 0
    while z1.size(1) > 1:
        if depth >= temporal_unit:
            loss += temporal_contrastive_loss(z1, z2)
        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    # Avoid division-by-zero when the input is too short for hierarchical pooling.
    return loss / max(depth, 1)


def temporal_contrastive_loss(z1, z2, temperature=0.1):
    B, T, _ = z1.size()
    if T == 1:
        return z1.new_tensor(0.)

    positive_pairs = torch.cat([z1, z2], dim=1)  # B x 2T x C
    positive_pairs = F.normalize(positive_pairs, dim=-1)  # keep dot-products bounded
    sim = torch.matmul(positive_pairs, positive_pairs.transpose(1, 2))  # B x 2T x 2T
    sim = sim / max(temperature, 1e-6)

    positive_logits = sim[:, :T, T:]
    log_probs = F.log_softmax(positive_logits, dim=-1)
    log_probs = torch.nan_to_num(log_probs, nan=0.0)

    loss = -log_probs.mean()
    return loss
