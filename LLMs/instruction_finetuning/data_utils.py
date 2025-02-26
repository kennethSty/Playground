from instruction_finetuning.InstructionDataset import InstructionDataset
from typing import Dict, List
import torch
from torch.utils.data import DataLoader

def split_data(data: List[Dict], train_frac = 0.85, test_frac = 0.1):
    train_end_idx = int(len(data) * train_frac)
    test_end_idx = train_end_idx + int(len(data) * test_frac)

    train_data = data[:train_end_idx]
    test_data = data[train_end_idx: test_end_idx]
    val_data = data[test_end_idx:] 

    return train_data, test_data, val_data


def set_up_loaders(train_data: List[Dict], test_data: List[Dict], val_data: List[Dict], tokenizer):

    num_workers = 0
    batch_size = 8
    
    train_dataset = InstructionDataset(train_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        collate_fn = custom_collate_fn,
        shuffle = True,
        drop_last = True,
        num_workers = num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        collate_fn = custom_collate_fn,
        shuffle = True,
        drop_last = False,
        num_workers = num_workers
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size = batch_size,
            collate_fn = custom_collate_fn,
            shuffle = True,
            drop_last = False,
            num_workers = num_workers
    )

    return train_loader, test_loader, val_loader

def custom_collate_fn(
        batch: torch.Tensor, 
        pad_token_id=50256, 
        device="mps",
        allowed_max_length=1024,
        ignore_index=-100):
        
    batch_max_length = max(len(item) + 1 for item in batch) #gets longest batch size + 1
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] #will be kept in target. Ensures target and input same length although target shifted
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        
        input_ids = torch.tensor(padded[:-1]) #tokens out the last inpout id
        target_ids = torch.tensor(padded[1:]) #target is inpout shifted by 1 index to the right)

        mask = (target_ids == pad_token_id)
        indices = torch.nonzero(mask).squeeze() #1d tensor of indices at which target == pad_token_id
        is_multiple_pad_ids = indices.numel() > 1

        if is_multiple_pad_ids:
            target_ids[indices[1:]] = ignore_index #replace EOS token with ignore index that wll be ignored by corss entroypy loss computation

        if allowed_max_length is not None:
            input_ids = input_ids[:allowed_max_length] #truncate to max seq len
            target_ids = target_ids[:allowed_max_length] 

        inputs_lst.append(input_ids) #list of tensors
        targets_lst.append(target_ids)

    inputs_tensor = torch.stack(inputs_lst).to(device) #stack -> one input sequence is one row
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

