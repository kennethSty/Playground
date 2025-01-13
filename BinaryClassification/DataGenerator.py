from CircleData import CircleData

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt 

class CircleDataGenerator: 

    RANDOM_SEED = 42
    BATCH_SIZE = 500
    __slots__ = ["n_train", "n_test", "noise", "train_data", "test_data", "train_loader", "test_loader", "device"]
    
    def __init__(self, n_train: int, n_test: int, noise: float, device: str):
        self.n_train = n_train
        self.n_test = n_test
        self.noise = noise
        self.device = device
        
    def generate_data(self):
        x_train, y_train = make_circles(self.n_train, noise = self.noise, random_state = self.RANDOM_SEED)
        x_test, y_test = make_circles(self.n_test, noise = self.noise, random_state = self.RANDOM_SEED)
        self.train_data = CircleData(x_train.squeeze(), y_train.squeeze(), self.device)
        self.test_data = CircleData(x_test.squeeze(), y_test.squeeze(), self.device)
        self.train_loader = DataLoader(self.train_data, self.BATCH_SIZE)
        self.test_loader = DataLoader(self.test_data, self.BATCH_SIZE)

    def show_data(self):
        plt.scatter(self.train_data.x[:, 0].cpu().detach(), self.train_data.x[:, 1].cpu().detach(), c=self.train_data.y.cpu().detach(), cmap=plt.cm.RdBu)
        plt.show()
         
                                
