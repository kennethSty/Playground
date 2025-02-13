from pathlib import Path
import tiktoken
import torch

from finetuning.finetuning_utils import setup_dataloaders
from finetuning.gpt_download import download_and_load_gpt2
from finetuning.load_model import load_weights_into_gpt
from gpt_architecture.GPTModel import GPTModel 
from gpt_architecture.text_generation_utils import generate, token_ids_to_text, text_to_token_ids

def main():
    MODEL_SIZE = "124M"
    MODEL_DIR = "gpt2"
    CHOOSE_MODEL = f"gpt2-small ({MODEL_SIZE})"
    N_CLASSES = 2
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12
    }

    #Setup dataloader
    train_ds_path = Path("preprocessing/datasets/train.csv")
    test_ds_path = Path("preprocessing/datasets/test.csv")
    val_ds_path = Path("preprocessing/datasets/validation.csv")
    train_loader, test_loader, val_loader = setup_dataloaders(
        batch_size=8,
        tokenizer=tokenizer,
        train_ds_path=train_ds_path,
        test_ds_path=test_ds_path,
        val_ds_path=val_ds_path
    )

    print("Loaded finetuning datasets")
    print("Training Batches:", len(train_loader))

    model = get_model_for_tuning(
        model_size=MODEL_SIZE,
        model_dir=MODEL_DIR,
        config=BASE_CONFIG,
        n_classes=N_CLASSES
    )

def get_model_for_tuning(model_size: int, model_dir: str, config: Dict[str], n_classes: int):    
    #Load OpenAI Model weights
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=model_dir
    )
    #Initialize model
    device = "mps" if torch.mps.is_available() else "cpu"
    model = GPTModel(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    #Load weights into model
    load_weights_into_gpt(model, params)
    #Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False #all params with this will not be optimized by optimizer
    #Replace last layer
    model.out_layer = torch.nn.Linear(
        in_features=config["emb_dim"],
        out_features=n_classes
    )
    #Make weights in last trf block and layer norm also trainable for finetuning
    for param in model.transformer_block[-1].parameters():
        param.requires_grad = True
    for param in model.layer_norm.parameters():
        param.requires_grad = True
    model.to(device)
    return model

if __name__ == "__main__":
    main()

