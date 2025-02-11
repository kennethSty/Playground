from gpt_architecture.GPTModel import GPTModel
from gpt_architecture.LayerNorm import LayerNorm
from gpt_architecture.TransformerBlock import TransformerBlock
from gpt_architecture.text_generation_utils import generate_next_token, text_to_token_ids, token_ids_to_text
from preprocessing.DataGenerator import DataGenerator
from preprocessing.GPTDatasetV1 import GPTDatasetV1
from gpt_architecture.config import GPT_CONFIG_124M as config
from pretraining.loss_utils import calc_loss_loader

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
    model = GPTModel().to(device)
    loss = calc_loss_loader(
        data_loader = data_generator.train_loader,
        model = model, 
        device = device
    )
    
    print("Loss", loss)
            
if __name__ == "__main__":
    main()    

 
    
