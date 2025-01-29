import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):

    def __init__(self, raw_text: str, tokenizer: tiktoken,Encoding, max_length: int, stride: int ):
        super().__init__()

        input_ids = []
        target_ids = []

        token_ids = tokenizer.encode(raw_text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.output_ids.append(output_chunk)

    def __getitem__(self, idx: int): 
        return self.input_ids[idx], self.output_ids[idx]

    def __len__(self):
        return len(self.input_ids)       
        
