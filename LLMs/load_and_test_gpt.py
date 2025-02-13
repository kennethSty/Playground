from finetuning.gpt_download import download_and_load_gpt2
from finetuning.load_model import load_weights_into_gpt
from gpt_architecture.config import GPT_CONFIG_124M
from gpt_architecture.GPTModel import GPTModel
from gpt_architecture.text_generation_utils import generate, token_ids_to_text, text_to_token_ids

import tiktoken
import torch

# Load OpenAI model keys
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
device = "mps" if torch.mps.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval();

load_weights_into_gpt(gpt, params)
gpt.to(device);

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
