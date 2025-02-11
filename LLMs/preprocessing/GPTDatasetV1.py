import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):

    def __init__(self, raw_text: str, tokenizer: tiktoken.Encoding, max_length: int, stride: int ):
        super().__init__()

        token_ids = tokenizer.encode(raw_text)
        self.input_ids = []
        self.target_ids = []
        self.vocab_dim = tokenizer.n_vocab

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = torch.tensor(token_ids[i : i + max_length])
            target_chunk = torch.tensor(token_ids[i + 1 : i + max_length + 1])
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __getitem__(self, idx: int): 
        return self.input_ids[idx], self.target_ids[idx]

    def __len__(self):
        return len(self.input_ids)       
        
