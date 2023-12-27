import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import Net
from train import train_model
from test import test_model
import matplotlib.pyplot as plt

#データの読み込み
train_dataset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = "./data", train = False, transform = transforms.ToTensor(), download = True)
fig, label = train_dataset[0]

#学習データ、テストデータの用意
batch_size = 256
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

input_size = 28 * 28
hidden1_size = 100
hidden2_size = 50
output_size = 10

#モデルの定義
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net(input_size, hidden1_size, hidden2_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)


train_loss_list = []
test_loss_list = []
train_accuracy_list = []
test_accuracy_list = []
num_epochs = 100

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device = device)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    test_loss, test_accuracy = test_model(model, test_loader, criterion, optimizer, device = device)
    test_loss_list.append(test_loss)
    test_accuracy_list.append(test_accuracy)
    
    print(f'epochs: {epoch}, train_loss: {train_loss:.5f}, test_loss: {test_loss:.5f}, train accuracy: {train_accuracy:.5f}, test accuracy: {test_accuracy:.5f}')

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

