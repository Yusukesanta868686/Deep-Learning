import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_init_features, num_classes):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, bias=False)
        self.norm0 = nn.BatchNorm2d(num_init_features)
        self.pool0 = nn.MaxPool2d(kernel_size=7, stride=2, padding=1)
        
        # Adjusted the input size of the fully connected layer
        # The input size should be based on the output size of the previous layers
        self.fc0 = nn.Linear(num_init_features * 128 * 128, num_classes)

    def forward(self, x):
        z1 = F.relu(self.norm0(self.conv0(x)))
        z2 = self.pool0(z1)
        #z2 = z1
        # Flatten the output before passing it to the fully connected layer
        z2 = z2.view(z2.size(0), -1)
        
        z3 = self.fc0(z2)
        return z3
