import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable, List
from torch.optim.lr_scheduler import StepLR

class ModelKeeper: 
    def __init__(self, 
                 model: nn.Module,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 train_loader: DataLoader,
                 test_loader: DataLoader):
                 
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, n_epochs: int, lr: float, weight_decay: float) -> List[float]:

        losses = []
        optimizer = torch.optim.Adam(lr = lr, params = self.model.parameters(),  weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size = 5, gamma=0.5) 
        self.model.train()
        
        for i in range(n_epochs): 
            epoch_sum = 0
            epoch_train_steps = 0
            for x, y in self.train_loader:
                self.model.zero_grad()
                y_pred_probs = self.model(x).squeeze()
                y_preds = torch.round(y_pred_probs)
                loss = self.loss_fn(y_preds, y)
                loss.backward()
                optimizer.step()
                losses.append(loss)
                accuracy = self.accuracy_fn(y_preds, y)
                epoch_sum += loss
                epoch_train_steps += 1
            self.log_gradients()
            print(f"Epoch: {i}, Accuracy {accuracy} Avg loss: {epoch_sum / epoch_train_steps}")
            #scheduler.step()

        return losses

    def test(self) -> float:
        self.model.eval()
        loss_sum = 0
        n_test_batches = 0
        with torch.inference_mode():
            for x, y in self.test_loader:
                y_pred = torch.round(self.model(x).squeeze())
                accuracy = self.accuracy_fn(y_preds, y)
                loss = self.loss_fn(y_pred, y)
                loss_sum += loss
                n_test_batches += 1
        return loss_sum / n_test_batches
    
    def accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred)) * 100 
        return acc
            
    def log_gradients(self):
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: Gradient norm = {param.grad.norm().item()}")
        
