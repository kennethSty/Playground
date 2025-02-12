from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DataGenerator:

    
    def __init__(self):
        BATCH_SIZE = 32
        self.train_data = datasets.FashionMNIST(
            root = "data",
            train = True,
            download = True,
            transform = ToTensor(),
            target_transform = None
        )

        self.test_data = datasets.FashionMNIST(
            root = "data",
            train = False,
            download = True, 
            transform = ToTensor(),
            target_transform = None
        )

        self.train_loader = DataLoader(self.train_data, batch_size = BATCH_SIZE, shuffle = True)
        self.test_loader = DataLoader(self.test_data, batch_size = BATCH_SIZE)

        print(f"Training data consists of {len(self.train_loader)} batches of {BATCH_SIZE}")
        print(f"Test data consists of {len(self.test_loader)} batches of {BATCH_SIZE}")

        

    def visualize_by_index(self, id: int, split: str):
        if split == "train":
            image, label = self.train_data[id]
        elif split == "test":
            image, label = self.train_data[id]
        else: 
            raise AttributeError             
        plt.imshow(image.squeeze(). cmap = "gray") #squeeze to get 1 [1, 28, 28] to [28, 28]
        plt.title(label)
        plt.show() 

   
