import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: str,
    eval_iter: int):

    total_loss = 0
    if eval_iter is None: 
        eval_iter = len(data_loader)
    else:
        eval_iter = min(eval_iter, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < eval_iter:    
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break

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

    #treat each token as a classification instance
    loss = nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten() #merges batch and token dim together 
    )
    return loss
    
