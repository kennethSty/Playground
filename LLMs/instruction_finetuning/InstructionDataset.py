import torch
from torch.utils.data import Dataset
import tiktoken
from typing import Dict
from instruction_finetuning.utils import format_input

class InstructionDataset(Dataset):
    def __init__(self, data: Dict, tokenizer: tiktoken.Encoding):
        self.data = data
        self.encoded_texts = []
        for entry in data: 
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, idx):
        return self.encoded_texts[idx]

    def __len__(self):
        return len(self.data)
