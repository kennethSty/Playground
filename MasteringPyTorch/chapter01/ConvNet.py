import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    Deep CNN for predicting mnist images.
    Input shape: (batch_size, 1 x28x28)
    """
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1), #out: (26, 26, 16),
             nn.ReLU(),
             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1), #out: (24, 24, 32)
             nn.ReLU(),
             nn.MaxPool2d(kernel_size = 2, stride = 2),
             nn.Dropout(0.1)                        
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 32, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)
