import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def train_model(model, train_loader, criterion, optimizer, device = 'cpu'):
    train_loss = 0
    num_train = 0
    train_correct = 0

    for i, (images, labels) in enumerate(train_loader):
        num_train += len(labels)

        images, labels = images.view(64, 3, 256*256, -1).to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)

        train_correct += (predicted == labels).sum().item()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        if i == 10: break
    train_loss = train_loss / num_train
    train_accuracy = train_correct / num_train

    return train_loss, train_accuracy