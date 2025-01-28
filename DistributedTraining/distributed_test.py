import os
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch import nn
import torch

from SomeData import SomeData
from NN import NN


def main(rank, world_size, num_epochs):
    torch.manual_seed(42)  #1
    ddp_setup(rank, world_size) #2

    train_loader, test_loader = prepare_dataset() #3
    loss_fn = nn.CrossEntropyLoss()
    model = NN(input_dim=2, output_dim=2, hidden_nodes=[4, 4])
    model.to(rank) #4
    model = DDP(model, device_ids=[rank]) #5
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  
    model.train()

    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            y_logits = model(features)
            loss = loss_fn(y_logits, labels)

            optimizer.zero_grad()
            loss.backward()  # 6
            optimizer.step()  

            print(f"[GPU{rank}] Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")

    print(f"GPU {rank} finished")
    destroy_process_group() #7


def prepare_dataset():
    train_ds = SomeData(torch.randn(2, 2), torch.randn(2))  # SomeData is a class extening Dataset
    test_ds = SomeData(torch.randn(2, 2), torch.randn(2))   

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=20,
        shuffle=False,  # 8
        pin_memory=True,  # 9
        drop_last=True,  # 10
        sampler=DistributedSampler(train_ds)  # 11
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=20,
        shuffle=False,  
        pin_memory=True,
        drop_last=False, 
        sampler=DistributedSampler(test_ds)  
    )

    return train_loader, test_loader

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # 12
    os.environ["MASTER_PORT"] = "1234"  # 13
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" #13!
    init_process_group(backend="gloo", rank=rank, world_size=world_size)  # 14
    if torch.mps.is_available():
        torch.mps.set_device(rank)  #15
    else:
        raise RuntimeError("mps backend not available")


if __name__ == "__main__":
    print("Numbe of GPUs available:", torch.mps.device_count())
    torch.manual_seed(42)
    num_epochs = 3
    world_size = torch.mps.device_count() 
    mp.spawn(main, args = (world_size, num_epochs), nprocs = world_size) #16
