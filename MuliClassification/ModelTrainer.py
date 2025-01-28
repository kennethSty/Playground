from torch import nn
import torch
from torch.utils.data import DataLoader
t
def train_model(model: nn.Module, lr: int, n_epochs: int, train_loader: DataLoader):
    optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    print("===============Start training the model===============")
    for epoch in range(n_epochs):
        epoch_loss_sum = 0
        epoch_training_steps = 0
        for x_batch, y_batch in train_loader:
            y_logits = model(x_batch)
            y_probs = torch.softmax(y_logits, dim = 1) #accross columns as colum values in one row are pred for eachlab
            y_pred = torch.argmax(y_probs, dim = 1)
            loss = loss_fn(y_logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss
            epoch_training_steps +=1
        print(f"Epoch: {epoch}, Avg. Loss {epoch_loss_sum / epoch_training_steps}")
            
        
def test_model(model: nn.Module, test_loader: DataLoader):

    model.eval()
    sum_correct_preds = 0
    sum_instances = 0

    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            y_logits = model(x_batch)
            y_probs = torch.softmax(y_logits, dim = 1)
            y_pred = torch.argmax(y_probs, dim = 1)
            sum_correct_preds += sum_correct_predictions(y_pred, y_batch)
            sum_instances += len(x_batch)

    print(f"Accuracy: {sum_correct_preds / sum_instances}")

def sum_correct_predictions(y_pred: torch.Tensor, y_true: torch.Tensor):
    sum = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            sum += 1
    return sum 
