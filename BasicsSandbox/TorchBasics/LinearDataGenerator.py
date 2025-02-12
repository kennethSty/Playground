import torch
import numpy as np
from torch.utils.data import DataLoader
from PreProcessingEngine import PreProcessingEngine
from utils import get_device
from RegDataSet import RegDataSet
class LinearDataGenerator():

    def __init__(self):
        self.device = get_device.get_device()

    def generate_data_loaders(self, start, end, number_instances, number_timesteps, test_size):
        
        x, y = self.generate_raw_data(
            start = start,
            end = end, 
            number_timesteps = number_timesteps,
            number_instances = number_instances
            )

        x_train, x_test, y_train, y_test = PreProcessingEngine(x, y).get_split(test_size)

        train_dataset = RegDataSet(x_train, y_train)
        test_dataset = RegDataSet(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size = 50)
        test_loader = DataLoader(test_dataset, batch_size = 50) 

        return train_loader, test_loader

        
    def generate_raw_data(self, start, end, number_instances, number_timesteps):

        x_total = []
        y_total = []
        
        for i in range(number_instances):
            bias = np.random.randn()
            weight = np.random.randn()
            x, y = self.generate_one_instance(start, end, weight, bias, number_timesteps)
            x_total.append(x)
            y_total.append(y)

        return torch.stack(x_total).to(self.device), torch.stack(y_total).to(self.device)    

    def generate_one_instance(self, start, end, weight, bias, number_timesteps):
        """
        Generate linear data based on the formula: y = weight * x + bias.
        """
        # Calculate step size based on the range and the number of instances
        step_size = (end - start) / number_timesteps
        x = torch.arange(start, end, step_size)  # Generate x values
        out_x = torch.cat((x, torch.tensor([bias, weight])), dim = -1) #pass in params as add input
        y = weight * x + bias  # Calculate y values based on the linear formula
        return out_x, y
    
