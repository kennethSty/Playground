import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet 

def train(model: nn.Module, device: str, train_dataloader: DataLoader, optimizer: optim, epochs: int):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            loss = F.cross_entropy(y_logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                accuracy = batch_accuracy(y_logits, y)
                print(f"Epoch: {epoch}, seen instances: {batch_idx * len(X)}/{len(train_dataloader) * len(X)}")
                print(f"Accuracy: {accuracy * 100} %")

def test(model: nn.Module, device: str, test_dataloader: DataLoader):
    model.eval()
    cumulative_batch_acc = 0
    num_batches = len(test_dataloader)
    with torch.inference_mode():
        for X, y in test_dataloader:
            X,y = X.to(device), y.to(device)
            y_logits = model(X)
            accuracy = batch_accuracy(y_logits, y)
            cumulative_batch_acc += accuracy

    print(f"Accuracy test set: {cumulative_batch_acc / num_batches}")


def batch_accuracy(y_logits: torch.Tensor, y: torch.Tensor):
    y_pred = torch.argmax(y_logits, -1)
    print(y_logits.shape)
    return torch.sum(y_pred == y).item() / len(y_pred)


def main():
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    
    train_dataloader = DataLoader(
        datasets.MNIST('../data', 
            train=True, 
            download = True,
            #normalizes input pixesl to have zero mean and unit variance -> better training
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)) #Tuple because normalize expects tuples even if it is only one element
            ])
        ),
        batch_size = 32, 
        shuffle = True
    )

    test_dataloader = DataLoader(
        datasets.MNIST('../data', 
            train = False, 
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,),(MNIST_STD,)),
            ])
        ),
        batch_size = 500,
        shuffle = False
   )
    
    torch.manual_seed(42)
    device = "mps"
    model = ConvNet().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)
    
    train(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        epochs = 2
    )
    
    test(
        model=model,
        device=device,
        test_dataloader=test_dataloader
    )

if __name__ == "__main__":
    main()
