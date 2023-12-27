import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(model, train_loader, criterion, optimizer, device = 'cpu'):
    train_loss = 0.0
    num_train = 0.0
    train_correct = 0
    for i, (images, labels) in enumerate(train_loader):
        #batch数をカウント
        num_train += len(labels)

        images, labels = images.view(-1, 28  * 28).to(device), labels.to(device)

        #勾配を初期化
        optimizer.zero_grad()

        #順伝播
        outputs = model(images)

        #損失の計算
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs, 1)

        # Calculate the number of correctly predicted labels
        train_correct += (predicted == labels).sum().item() 

        #誤差の逆伝播
        loss.backward()

        #パラメータの更新
        optimizer.step()

        #lossを加算
        train_loss += loss.item()

    #lossの平均を取る 
    train_loss = train_loss / num_train
    train_accuracy = train_correct / num_train
    return train_loss, train_accuracy