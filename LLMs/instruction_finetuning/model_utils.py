import sys
import os
import torch
import tiktoken
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add LLMs directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add parent directory
from finetuning.gpt_download import download_and_load_gpt2
from finetuning.load_model import load_weights_into_gpt
from gpt_architecture.GPTModel import GPTModel 
from gpt_architecture.text_generation_utils import generate, token_ids_to_text, text_to_token_ids

def set_up_model():
    MODEL_SIZE = "355M"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16
    }

    #Load OpenAI Model weights
    settings, params = download_and_load_gpt2(
        model_size=MODEL_SIZE, models_dir="gpt2"
    )
    #Initialize model
    device = "mps" if torch.mps.is_available() else "cpu"
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    return model.to(device)

if __name__ == "__main__":
    set_up_model()
