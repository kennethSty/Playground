from gpt_architecture.GPTModel import GPTModel
from gpt_architecture.LayerNorm import LayerNorm
from gpt_architecture.TransformerBlock import TransformerBlock
from gpt_architecture.text_generation_utils import generate_next_tokens, text_to_token_ids, token_ids_to_text
from preprocessing.DataGenerator import DataGenerator
from preprocessing.GPTDatasetV1 import GPTDatasetV1
from gpt_architecture.config import GPT_CONFIG_124M as config
from pretraining.loss_utils import calc_loss_loader
from pretraining.pretraining import train_model

import tiktoken
import torch.nn as nn
import torch


def read_text(filename: str) -> str:
    with open(filename, "r", encoding = "utf-8") as f:
        raw_text = f.read()
        print("Length of text:", len(raw_text))
    return raw_text

def main():

    file_name = "preprocessing/the-verdict.txt"
    raw_text = read_text(file_name)
    tokenizer = tiktoken.get_encoding("gpt2")

    data_generator = DataGenerator(
        batch_size = 2,
        max_length = config["context_length"],
        stride = config["context_length"],
        drop_last = False,
        shuffle = False,
        num_workers = 0
    )
    
    data_generator.setup_data_loaders(
        raw_text = raw_text,
        train_ratio = 0.9 )

    device = "mps" if torch.mps.is_available() else "cpu"
    print("Device:", device)
    model = GPTModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
     
    num_epochs = 1
    eval_freq = 5
    start_context = "every effort moves you"
    
    train_losses, val_losses, tokens_seen = train_model(
        model = model,
        train_loader = data_generator.train_loader,
        val_loader = data_generator.test_loader,
        optimizer = optimizer,
        device = device,
        num_epochs = num_epochs,
        eval_freq = eval_freq,
        start_context = start_context,
        tokenizer = tokenizer   
    )

    print("Saving checkpoint")

    torch.save(
        {
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict()
        }, 
        "model_and_optimizer.pth"
    )

    print("loading checkpoint")

    checkpoint = torch.load('model_and_optimizer.pth', map_location = device)
    model = GPTModel().to(device)
    model.load_state_dict(checkpoint["model_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
    optimizer.load_state_dict(checkpoint["optimizer_dict"])        

    print("train again")
    
    train_losses, val_losses, tokens_seen = train_model(
        model = model,
        train_loader = data_generator.train_loader,
        val_loader = data_generator.test_loader,
        optimizer = optimizer,
        device = device,
        num_epochs = num_epochs,
        eval_freq = eval_freq,
        start_context = start_context,
        tokenizer = tokenizer   
    )

    
if __name__ == "__main__":
    main()    

 
    
