from pretraining.loss_utils import calc_loss_batch, calc_loss_loader
from gpt_architecture.text_generation_utils import generate_next_tokens, text_to_token_ids, token_ids_to_text

import torch.nn as nn
from torch.utils.data import DataLoader
import torch 
import tiktoken

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int, 
    start_context: str,
    tokenizer: tiktoken.Encoding):
    """Trains the model"""

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): ")
                print(f"Train loss {train_loss:.3f}")
                print(f"Val loss {val_loss:.3f}")

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    eval_iter: int):
    """Evaluates if the model improves during training"""

    model.eval()
    with torch.inference_mode():
        train_loss = calc_loss_loader(
            train_loader, model, device, eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, eval_iter
        )
        
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    device: str,
    start_context:str):
    """Prints a text continued from start_context by the model"""
    
    model.eval()
    context_size = model.pos_embed_layer.weight.shape[0]
    print("context size", context_size)

    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.inference_mode():
        generated_ids = generate_next_tokens(
            model=model, token_ids=encoded,
            max_new_tokens=50,context_size=context_size
        )
    decoded_text = token_ids_to_text(generated_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
        

