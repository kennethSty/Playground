import json
import os
import urllib.request
import torch
from torch.utils.data import DataLoader
from typing import List, Dict

def download_and_load_file(file_path: str, url: str):

    if not os.path.exists(file_path):
    
        #open url and write text into filepath file
        with urllib.request.urlopen(url) as response:
            url_text = response.read().decode("utf-8")
        with open(file_path, "w", encoding = "utf-8") as f:
            print(url_text)
            f.write(url_text)

    else:
        with open(file_path, "r", encoding = "utf-8") as f:
            text_data = f.read()

    with open(file_path, "r") as f:
        data = json.load(f)

    return data


def format_input(entry: str):

    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request. "
        f"\n\n### Instruction: \n{entry['instruction']}"  
    )

    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
    )

    return instruction_text + input_text
