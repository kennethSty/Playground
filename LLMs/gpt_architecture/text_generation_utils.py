from gpt_architecture.GPTModel import GPTModel 

import tiktoken
import torch


def generate_next_token(
        model: GPTModel, 
        token_ids: torch.Tensor, 
        max_new_tokens: int, context_size: int):

    batch_size, n_tokens = token_ids.shape
    
    for _ in range(max_new_tokens):
        context_token_ids = token_ids[:, -context_size:] #choose context_size most recent tokens
        with torch.inference_mode():
            logits = model(context_token_ids)
            batch_size, n_tokens, vocab_size = logits.shape
            next_word_logits = logits[:, -1, :] #nwl.shape (batch, vocab_size)
            next_word_probs = torch.softmax(next_word_logits, dim = -1) #shape: (batch, vocab_size)
            next_word_id = torch.argmax(next_word_probs, dim = -1, keepdim = True) #shape (batch, 1)
            token_ids = torch.cat([token_ids, next_word_id], dim = -1)
        
    return token_ids    


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
