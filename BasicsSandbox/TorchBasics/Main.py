import torch
from torch.utils.data import DataLoader
from LinearDataGenerator import LinearDataGenerator
from RegressionNet import RegNet
from RegDataSet import RegDataSet
from ModelTrainer import ModelTrainer
from utils import get_device
def main():

    #generate data
    start = 0
    end = 10
    number_timesteps = 10
    number_linear_params = 2 #bias and weight
    input_size = number_timesteps + number_linear_params
    output_size = number_timesteps
    number_instances = 10000
    test_size = 0.2
    lr = 0.0001
    optimizer_name = "Adam"
    weight_decay = 0.05
    epochs = 10
    device = get_device.get_device()

    generator = LinearDataGenerator()
    
    train_loader, test_loader = generator.generate_data_loaders(
        start = start,
        end = end, 
        number_instances = number_instances,
        number_timesteps = number_timesteps,
        test_size = test_size
    )

    model = RegNet(input_size, output_size, [128, 256, 128]).to(device)
    trainer = ModelTrainer(model = model, 
        train_loader = train_loader, 
        optimizer_name = optimizer_name, 
        lr = lr, 
        weight_decay = weight_decay)
    model, losses = trainer.train(10)

if __name__ == "__main__": #Note if this file is run via Main.py directly then __name__ is __main__
    main()
