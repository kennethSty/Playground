import torch
import tiktoken
from GPTModel import GPTModel
from LayerNorm import LayerNorm
from TransformerBlock import TransformerBlock
from config import GPT_CONFIG_124M as config
 
def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text = "Every now and then I do"
    text2 = "Also sometimes I cannot help but"

    batch.append(torch.tensor(tokenizer.encode(text)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim = 0) #make batch dimension first one
    print(batch.shape)
    
    gpt = GPTModel()
    next_tokens = generate_next_token(
        model = gpt,
        token_ids = batch,
        max_new_tokens = 2,
        context_size = 4
    )
    decoded_text = tokenizer.decode(next_tokens.squeeze(0).tolist())
    print(decoded_text)
    print(batch.shape)
    print(out.shape)

    ln = LayerNorm(emb_dim = 5)
    batch_example = torch.randn(1,2,5)
    normalized = ln(batch_example)

    mean_norm = normalized.mean(dim = -1, keepdim = True)
    var_norm = normalized.var(dim = -1, keepdim = True)

def generate_next_token(model: GPTModel, token_ids: torch.Tensor, max_new_tokens: int, context_size: int):
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
                

if __name__ == "__main__":
    main()
