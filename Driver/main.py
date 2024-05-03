import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.utils.data import random_split
import pandas as pd
import os
from load_images import CustomDataset
#from densenet import DenseNet
from tinymodel import Net
from resnet import ResNet18
from train import train_model
from test import test_model
import matplotlib.pyplot as plt

def main():
    train_dir = "state-farm-distracted-driver-detection/imgs/train"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    total_dataset = CustomDataset(train_dir, transform = transform, mode = 'train')
    train_size = int(0.8 * len(total_dataset))
    val_size = len(total_dataset) - train_size

    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size]) 
    
    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = DenseNet(num_classes=10)
    #model = models.densenet169
    #model = Net(64, 10)
    model = ResNet18(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    num_epochs = 10000
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device = device)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss, test_accuracy = test_model(model, val_loader, criterion, optimizer, device = device)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        
        print(f'epochs: {epoch}, train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}, train accuracy: {train_accuracy:.5f}, test accuracy: {test_accuracy:.5f}')
        #print(f'epochs: {epoch}, train_loss: {train_loss:.5f}, train_accuracy: {train_accuracy:.5f}')
    # Create figure and axis objects with a shared x-axis
    fig, ax1 = plt.subplots()


    # Loss (left y-axis)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(train_loss_list, label='Train Loss', color='r')
    ax1.plot(test_loss_list, label='Test Loss', color='g')
    ax1.tick_params(axis='y')

    # Make the y-axis label, ticks and tick labels match the line color.
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('accuracy')  # we already handled the x-label with ax1
    ax2.plot(train_accuracy_list, label='Train Accuracy', color='r')
    ax2.plot(test_accuracy_list, label='Test Accuracy', color='g')
    ax2.tick_params(axis='y')

    # Adding legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Show plot
    plt.title('Training and Testing Loss and Accuracy over Epochs')
    plt.savefig('result/result.png')
    plt.show()

main()




