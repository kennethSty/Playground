import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str):

    total_loss = 0
    for input_batch, target_batch in data_loader:
        print
        loss = calc_loss_batch(
            input_batch, target_batch, model, device
        )
        total_loss += loss.item()

    return total_loss/len(data_loader)
    


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: str) -> float:

    """Computes loss for a single batch """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss
    
