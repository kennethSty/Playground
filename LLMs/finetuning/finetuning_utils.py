from pathlib import Path
import tiktoken
import sys
import os
from torch.utils.data import DataLoader
# Add the parent directory (LLMs) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.SpamDataset import SpamDataset

def setup_dataloaders(
    batch_size: int, 
    tokenizer: tiktoken.Encoding,
    train_ds_path: str, 
    test_ds_path:str , 
    val_ds_path: str):

    train_dataset = SpamDataset(
        csv_file=train_ds_path, tokenizer=tokenizer
    )    
    test_dataset = SpamDataset(
        csv_file=test_ds_path, tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=val_ds_path, tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return train_loader, test_loader, val_loader
