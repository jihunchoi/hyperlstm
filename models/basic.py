import torch
from torch.nn import functional


def sequence_cross_entropy(logits, targets, masks=None):
    log_probs = functional.log_softmax(input=logits, dim=2)
    targets = targets.unsqueeze(2)
    losses = -torch.gather(log_probs, dim=2, index=targets)
    denom = targets.numel()
    if masks:
        masks = masks.unsqueeze(2)
        losses = losses * masks
        denom = masks.sum()
    loss = losses.sum() / denom
    return loss
