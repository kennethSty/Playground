import torch

class get_device:

    def __init__(self):
        pass
    
    def get_device():
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        return device
        
