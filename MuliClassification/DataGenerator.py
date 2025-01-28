from BlobData import BlobData

from sklearn.datasets import make_blobs
from torch.utils.data import DataLoader
import torch


class DataGenerator:
    def __init__(self, batch_size, n_train, n_test, device, num_features = 2, num_classes = 4, cluster_std = 1.5 , random_seed = 42):

        X_train, y_train = make_blobs(n_samples=n_train,
            n_features = num_features, 
            centers = num_classes, 
            cluster_std = cluster_std, 
            random_state = random_seed
        )

        X_test, y_test = make_blobs(n_samples=n_test,
            n_features = num_features, 
            centers = num_classes, 
            cluster_std = cluster_std, 
            random_state = random_seed
        )

     
        self.train_data = BlobData(torch.Tensor(X_train), torch.Tensor(y_train), device = device)
        self.test_data = BlobData(torch.Tensor(X_test), torch.Tensor(y_test), device = device)
        self.train_loader = DataLoader(self.train_data, batch_size = batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size = batch_size)
