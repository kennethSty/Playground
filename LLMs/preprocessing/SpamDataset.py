import torch
from torch.utils.data import Dataset
import tiktoken
import pandas as pd

class SpamDataset(Dataset):

    def __init__(self, csv_file:str, 
        tokenizer:tiktoken.Encoding, 
        padding_text="<|endoftext|>"):

        pad_token_id = tokenizer.encode(padding_text, allowed_special={"<|endoftext|>"})

        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]        
        self.max_length = self._longest_encoded_length()
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (
                self.max_length - len(encoded_text)
            )
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label1"]
        return (
            torch.tensor(endcoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length

        return max_length
