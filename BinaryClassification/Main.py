from DataGenerator import CircleDataGenerator
from ModelKeeper import ModelKeeper
from CircleDetector import CircleDetector
from Plotter import Plotter

import torch

def main(): 

    device = get_device()
    n_train = 100000 
    n_test = 100 
    noise = 0
    print(f"==========Using device: {device}==========")
    print(f"==========Using n_train: {n_train}==========")
    print(f"==========Using n_test: {n_test}==========")
    print(f"==========Using noise: {noise}==========")
    
    print("==========Start generating data==========")
    data_generator = CircleDataGenerator(n_train = n_train, n_test = n_test, noise = noise, device = device)
    data_generator.generate_data()
    print("==========Finished generating data==========")

    print(f"==========Start training the model==========")
    model = CircleDetector(input_dim = 2, output_dim = 1, hidden_layers = [4,4])
    model.initialize_weights()
    model = model.to(device)
    loss_fn =  torch.nn.BCELoss()
    model_keeper = ModelKeeper(model = model, 
                               loss_fn = loss_fn, 
                               train_loader = data_generator.train_loader,
                               test_loader = data_generator.test_loader)
    loss_values_train = model_keeper.train(n_epochs = 50, lr = 0.2, weight_decay = 0)
    loss_values_test = model_keeper.test()
    print(f"==========Finish training the model==========")

    plotter = Plotter(loss_values_train)
    plotter.plot_convergence()
    

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.mps.is_available():
        return "mps"
    return "cpu"

if __name__ == "__main__":
    main()
