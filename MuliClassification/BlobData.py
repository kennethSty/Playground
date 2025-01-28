from torch.utils.data import Dataset, DataLoader
import torch 

class BlobData(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str):
        self.x = x.to(device)
        self.y = y.to(device)
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, id):
        return self.x[id], self.y[id]
        
            
