import torch
import torch.nn as nn
import torch.nn.functional as F

def test_model(model, test_loader, criterion, optimizer, device = 'cpu'):
    test_loss = 0.0
    num_test = 0
    test_correct = 0

    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            print(i)
            num_test += len(labels)
            images, labels = images.view(64, 3, 256 * 256, -1).to(device), labels.to(device)

            #推論
            outputs = model(images)

            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()

            if i == 10: break
        test_loss = test_loss / num_test
        test_accuracy = test_correct / num_test 
    return test_loss, test_accuracy