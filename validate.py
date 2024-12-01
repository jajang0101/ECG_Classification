import numpy as np
import torch
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

testFiles = ["9", "10"]

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
dataset = TimeSeriesDataset(x_test, y_test)
dataloader = DataLoader(dataset)

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
model.load_state_dict(torch.load("model_trained.pth"))
model.eval()
device = 'cuda'
model.to(device)

predictions = torch.empty(list(y_test.size())[0], 4)
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        predictions = (outputs > 0.5).int()
        #print(predictions)
'''
confusion = torch.zeros(4, 4)
instance_i = 0
for instance in enumerate(predictions):
    for i in range(4):
        confusion[i] = y_test[instance_i][i] + instance[i]
    instance_i += 1
instance_i = 0
for instance in enumerate(predictions):
    for i in range(4):
        confusion[instance_i][i] = y_test[instance_i][i] + instance[i]
    instance_i += 1 

print(confusion)
scpdict = {
    0: "MI",
    1: "STTC",
    2: "CD",
    3: "HYP"
}
scpCodes = ["MI", "STTC", "CD", "HYP"]

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + scpCodes, rotation=90)
ax.set_yticklabels([''] + scpCodes)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
'''
