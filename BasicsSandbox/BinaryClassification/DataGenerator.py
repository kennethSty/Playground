from CircleData import CircleData

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt 
import requests
from pathlib import Path 
from helper_functions import plot_predictions, plot_decision_boundary                                

class CircleDataGenerator: 

    RANDOM_SEED = 42
    BATCH_SIZE = 1000
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

    def download_helpers(self):         
        # Download helper functions from Learn PyTorch repo (if not already downloaded)
        if Path("helper_functions.py").is_file():
          print("helper_functions.py already exists, skipping download")
        else:
          print("Downloading helper_functions.py")
          request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
          with open("helper_functions.py", "wb") as f:
            f.write(request.content)
        

    def plot_decision_boundary(self, model):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, self.train_data.x.to(self.device), self.train_data.y.to(self.device))
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, self.test_data.x.to(self.device), self.test_data.y.to(self.device))
