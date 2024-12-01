import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  # Shape: (num_samples, 12, 1000)
        self.labels = labels  # Shape: (num_samples, 4) - 4 binary labels per sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # Shape: (12, 1000)
        y = self.labels[idx]  # Shape: (4,)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

trainFiles = ["1", "2", "3", "4"]
testFiles = ["9", "10"]

x_train = np.empty((1, 12, 1000), dtype=np.float16)
y_train = np.empty((1, 4), dtype=np.uint8)

for file_name in trainFiles:
    with open(file_name + '.npy', 'rb') as saveFile:
        loaded = pickle.load(saveFile)
        x_train = np.append(x_train, loaded[0], axis = 0)
        y_train = np.append(y_train, loaded[1], axis = 0)

x_train = np.delete(x_train, 0 ,0)
y_train = np.delete(y_train, 0 ,0)
x_train = torch.tensor(x_train, dtype=torch.float16)
y_train = torch.tensor(y_train, dtype=torch.torch.uint8)

#model = torch.nn.RNN(12000, 128, 3)
#output, hn = model(x_train)

dataset_train = TimeSeriesDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)

x_test = np.empty((1, 12, 1000), dtype=np.float16)
y_test = np.empty((1, 4), dtype=np.uint8)

for file_name in testFiles:
    with open(file_name + '.npy', 'rb') as saveFile:
        loaded = pickle.load(saveFile)
        x_test = np.append(x_test, loaded[0], axis = 0)
        y_test = np.append(y_test, loaded[1], axis = 0)

x_test = np.delete(x_test, 0 ,0)
y_test = np.delete(y_test, 0 ,0)
x_test = torch.tensor(x_test, dtype=torch.float16)
y_test = torch.tensor(y_test, dtype=torch.torch.uint8)
dataset_test = TimeSeriesDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test)

class BinaryClassificationRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=4):
        super(BinaryClassificationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # RNN/GRU can also be used
        self.fc = torch.nn.Linear(hidden_size, num_classes)  # Maps hidden states to 4 binary outputs
        self.sigmoid = torch.nn.Sigmoid()  # For binary classification

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # RNN Forward
        out, _ = self.rnn(x, (h0, c0))  # out: (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]  # Use the last hidden state for classification

        # Fully connected and sigmoid activation
        out = self.fc(out)
        out = self.sigmoid(out)  # Each output between 0 and 1
        return out

model = BinaryClassificationRNN(input_size=1000, hidden_size=128, num_layers=2, num_classes=4)
model.to('cuda')

device = 'cuda'
criterion = torch.nn.BCELoss()  # Binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
lc_x = np.linspace(1, num_epochs, num=num_epochs)
lc_y = np.empty(num_epochs)
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in dataloader_train:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lc_y[epoch] = loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "model_trained.pth")
print("Model saved to model_trained.pth")

plt.plot(lc_x, lc_y)
plt.xlabel('Epochs')
plt.ylabel('BCELoss')
plt.show()

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print()

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])
