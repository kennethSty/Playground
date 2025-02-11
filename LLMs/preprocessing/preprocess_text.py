from tokenizing_text.DataGenerator import DataGenerator
from tokenizing_text.GPTDatasetV1 import GPTDatasetV1
import torch.nn as nn
import torch
from gpt_architecture.config import GPT_CONFIG_124M as config

def read_text(filename: str) -> str:
    with open(filename, "r", encoding = "utf-8") as f:
        raw_text = f.read()
        print("Length of text:", len(raw_text))
    return raw_text

def main():

    file_name = "tokenizing_text/the-verdict.txt"
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
    
    print("Train loader")
    for x,y in data_generator.train_loader:
        print(x.shape, y.shape)

    print("Test loader")
    for x,y in data_generator.test_loader:
        print(x.shape, y.shape)    
        
if __name__ == "__main__":
    main()    
