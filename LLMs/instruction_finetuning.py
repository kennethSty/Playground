from instruction_finetuning.utils import download_and_load_file, format_input
from instruction_finetuning.data_utils import split_data, set_up_loaders
from instruction_finetuning.model_utils import set_up_model
from pretraining.loss_utils import calc_loss_loader
from pretraining.pretraining import train_model
from gpt_architecture.text_generation_utils import generate, token_ids_to_text, text_to_token_ids
import torch
import tiktoken 
import time

def main():
    file_path = "instruction_data.json"
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json")
    data = download_and_load_file(file_path, url)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = "mps" if torch.mps.is_available else "cpu"
        
    train_data, test_data, val_data = split_data(data) 
    train_loader, test_loader, val_loader = set_up_loaders(train_data, test_data, val_data, tokenizer)

    model = set_up_model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.00005, weight_decay=0.1
    )
    num_epochs = 2

    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context = format_input(val_data[0]), tokenizer = tokenizer
    )
    end_time = time.time()
    execution_minutes = (end_time - start_time) / 60
    print(f"Finished training in {execution_minutes} minutes")

    
if __name__ == "__main__":
    main()    
