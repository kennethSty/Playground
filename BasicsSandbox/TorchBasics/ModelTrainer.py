import torch
from torch import nn

class ModelTrainer:
    def __init__(self, model, train_loader, optimizer_name, lr, weight_decay = 0):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = nn.MSELoss()
        if optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        else:
            self.optimizer - torch.optim.SGD(model.parameters(), lr = lr, weight_decay = weight_decay)

    def train(self, epochs):
        losses = []
        self.model.train()
        for epoch in range(epochs):
            for x_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss)         
            print(f"Epoch: {epoch}, Loss: {loss}")       
        return self.model, losses
