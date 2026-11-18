"""Preserve the absolute-coefficient objective while projecting one conflicting source pair."""

import torch


def source_losses(prediction, targets, std, sources):
    """Source contributions sum to the original masked Cd MAE + Cl MAE."""
    observed = torch.isfinite(targets)
    clean = torch.where(observed, targets, 0)
    errors = abs((prediction - clean) * std)
    weights = observed / observed.sum(0).clamp_min(1)
    per_case = (errors * weights).sum(1)
    return {int(s): per_case[sources == s].sum() for s in torch.unique(sources)}


def project_pair(first, second):
    """Symmetric two-source PCGrad step using the original opposing gradients."""
    dot = first @ second
    if float(dot) >= 0:
        return first, second
    return (
        first - dot / second.square().sum().clamp_min(1e-12) * second,
        second - dot / first.square().sum().clamp_min(1e-12) * first,
    )


def backward(losses, shared, pair=None):
    """Other parameters and sources retain their ordinary gradients."""
    update, record = None, None
    if pair is not None:
        vectors = [
            torch.cat(
                [
                    g.flatten()
                    for g in torch.autograd.grad(losses[s], shared, retain_graph=True)
                ]
            )
            for s in pair
        ]
        first, second = project_pair(*vectors)
        before = (vectors[0] @ vectors[1]) / (
            vectors[0].norm() * vectors[1].norm()
        ).clamp_min(1e-12)
        after = (first @ second) / (first.norm() * second.norm()).clamp_min(1e-12)
        record = {
            "pair": list(pair),
            "cosine_before": float(before),
            "cosine_after": float(after),
            "projected": bool(before < 0),
        }
        if record["projected"]:
            update = first + second - vectors[0] - vectors[1]
    sum(losses.values()).backward()
    if update is not None:
        start = 0
        for parameter in shared:
            count = parameter.numel()
            parameter.grad.add_(update[start : start + count].reshape_as(parameter))
            start += count
    return record


@torch.no_grad()
def update_average(average, model, decay):
    """GroupNorm has no running statistics; all stored feature moments stay fixed."""
    destination, source = list(average.parameters()), list(model.parameters())
    torch._foreach_mul_(destination, decay)
    torch._foreach_add_(destination, source, alpha=1 - decay)
