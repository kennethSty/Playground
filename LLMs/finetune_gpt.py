from pathlib import Path
import tiktoken
import torch

from finetuning.finetuning_utils import setup_dataloaders, calc_accuracy_loader, calc_loss_loader, finetune_model
from finetuning.gpt_download import download_and_load_gpt2
from finetuning.load_model import load_weights_into_gpt
from gpt_architecture.GPTModel import GPTModel 
from gpt_architecture.text_generation_utils import generate, token_ids_to_text, text_to_token_ids

def main():
    MODEL_SIZE = "124M"
    CHOOSE_MODEL = f"gpt2-small ({MODEL_SIZE})"
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

    #Load OpenAI Model weights
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="gpt2"
    )
    #Initialize model
    device = "mps" if torch.mps.is_available() else "cpu"
    model = GPTModel(BASE_CONFIG)
    tokenizer = tiktoken.get_encoding("gpt2")

    #Load weights into model, freeze params and add classification head
    load_weights_into_gpt(model, params)
    for param in model.parameters():
        param.requires_grad == False
    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad == True
    for param in model.layer_norm.parameters():    
        param.requires_grad == True
    model.out_layer = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=2
    )
    model.to(device)

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

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=5e-5,
        weight_decay=0.1
    )
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = finetune_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
        eval_freq=50,
        eval_iter=5
    )

if __name__ == "__main__":
    main()

