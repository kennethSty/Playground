from DataGenerator import DataGenerator
from MultiNet import MultiNet
from ModelTrainer import train_model, test_model

import torch

def main():
    device = get_device() 
    data = DataGenerator(batch_size = 20, n_train = 5000, n_test = 200, num_features = 2, num_classes = 4, device = device)
    model = MultiNet(input_features = 2, output_features = 4, hidden_nodes = 8).to(device)
    train_model(model = model, lr = 0.01, n_epochs = 10, train_loader = data.train_loader)
    test_model(model = model, test_loader = data.test_loader)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
