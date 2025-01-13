import torch
from torch.utils.data import Dataset

class CircleData(Dataset):

    def __init__(self, x, y, device):
        self.x = torch.Tensor(x).to(device) 
        self.y = torch.Tensor(y).to(device)
        
    def __getitem__(self, id):
        return self.x[id], self.y[id]  

    def __len__(self):
        return len(self.x)

