from torch.utils.data import DataLoader
from preprocessing.GPTDatasetV1 import GPTDatasetV1
import tiktoken

class DataGenerator:
    def __init__(self, batch_size = 8, max_length = 4, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
        self.batch_size = batch_size 
        self.max_length = max_length 
        self.stride = stride 
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        
    def setup_data_loaders(self, raw_text: str, train_ratio: float):

        tokenizer = tiktoken.get_encoding("gpt2")
        splid_idx = int(train_ratio * len(raw_text))
        train_text = raw_text[:splid_idx]
        test_text = raw_text[splid_idx:]
        
        train_ds = GPTDatasetV1(
            raw_text = train_text, 
            tokenizer = tokenizer, 
            max_length = self.max_length, 
            stride = self.stride)

        test_ds = GPTDatasetV1(
            raw_text = test_text, 
            tokenizer = tokenizer, 
            max_length = self.max_length, 
            stride = self.stride)
        
            
        self.vocab_dim = tokenizer.n_vocab
        self.train_loader = DataLoader(
            train_ds,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last = self.drop_last,
            num_workers = self.num_workers
        )        
        self.test_loader = DataLoader(
            test_ds,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last = self.drop_last,
            num_workers = self.num_workers
        )
        
    
