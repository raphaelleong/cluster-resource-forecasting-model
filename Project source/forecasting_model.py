#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
from preprocessing import *

# Parameters
train_size = 720
batch_size = 60
valid_size = 120
test_size = 1380

# tau
input_size = 120

# size of hidden layers
learning_rate = 0.00059
decay = 0.4
num_epochs = 500
dtype = torch.float

faster_sampling = False


# Model
class LSTM(nn.Module):

    def __init__(self, input_dim, batch_size):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.channels = int(batch_size / input_dim)

        # Conv1d with lstm
        self.conv1 = nn.Conv1d(in_channels=self.channels, out_channels=self.channels,
                               kernel_size=4, padding=1)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.lstm_x = nn.LSTM(self.input_dim, 128, 1)
        self.lstm_y = nn.LSTM(128, 64, 1)

        # Stacked LSTM
        self.lstm1 = nn.LSTM(self.input_dim, 128, 1)
        self.lstm2 = nn.LSTM(128, 64, 1)

        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        # ignoring hidden states for it to be stateful
        # CNN LSTM
        conv_out = self.conv1(x)
        max_out = self.pooling(conv_out)
        lstm_out, self.hidden = self.lstm_x(max_out)
        lstm_out, self.hidden = self.lstm_y(lstm_out)
        out = self.linear(lstm_out)

        # Standard LSTM
        # lstm_out, self.hidden = self.lstm1(x)
        # lstm_out, self.hidden = self.lstm2(lstm_out)
        # out = self.linear(lstm_out)

        return out.view(-1)


# input_acc = [2, 5, 6, 10, 15, 20, 30]

machine_acc = [348805021, 350588109, 4820285492, 1436333635, 3338000908, 1390835522, 1391018274, 5015788232, 4874102959]

mse_acc = 0
mse_res = []
mae_acc = 0

validation = True

# for input_var in input_acc:
#     input_size = input_var
for m_id in machine_acc:

    # Data pre processing
    data_train, data_valid, data_test = load_data_from_csv(m_id, train_size, valid_size, test_size,
                                                           input_size + 1, faster_sampling)

    data_train = torch.from_numpy(data_train).type(torch.Tensor)
    data_valid = torch.from_numpy(data_valid).type(torch.Tensor)
    data_test = torch.from_numpy(data_test).type(torch.Tensor)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
    if validation:
        dataloader_test = torch.utils.data.DataLoader(data_valid, batch_size=batch_size)
    else:
        dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

    model = LSTM(input_size, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training
    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        for batch_idx, batch_sample in enumerate(dataloader_train):
            bs = batch_sample[:, :input_size]

            X_train = bs.view([input_size, -1, input_size])
            y_target = batch_sample[:, input_size:]

            optimizer.zero_grad()
            model.train()

            # Forward pass
            y_pred = model(X_train)
            loss = loss_fn(y_pred, y_target)

            epoch_loss = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

        if t % 100 == 0:
            print("Epoch ", t, "MSE: ", epoch_loss)
        hist[t] = epoch_loss
    print("Epoch ", t, "MSE: ", epoch_loss)
    print("====================")

    # Testing evaluation
    model.eval()

    pred = None
    target = None

    for batch_idx, batch_sample in enumerate(dataloader_test):
        bs = batch_sample[:, :input_size]
        X_train = bs.view([input_size, -1, input_size])
        y_target = batch_sample[:, input_size:]

        y_pred = model(X_train)
        if pred is None:
            pred = y_pred
            target = y_target
        else:
            pred = torch.cat([pred, y_pred])
            target = torch.cat([target, y_target])

    if validation:
        test_rmse = loss_fn(pred, target).item()
    else:
        test_rmse = np.sqrt(loss_fn(pred, target).item())
    test_mae = L1_loss(pred, target).item()

    mse_acc += test_rmse
    mae_acc += test_mae
    # mse_res.append(test_rmse)

    if validation:
        print("Machine ID: ", m_id, "MSE: ", test_rmse)
    else:
        print("Machine ID: ", m_id, "RMSE: ", test_rmse, "MAE: ", test_mae)
    print("====================")

    if validation:
        plot_size = valid_size
    else:
        plot_size = test_size

    x_time = []
    if faster_sampling:
        for t in range(0, plot_size):
            x_time.append(t)
    else:
        for t in range(0, plot_size * 5, 5):
            x_time.append(t)

    # Evaluation

    # Predictions vs Target
    plt.plot(x_time, pred.detach().numpy(), label="Predictions")
    plt.plot(x_time, target.detach().numpy(), label="Data")
    plt.xlabel('Time shift (min)')
    plt.ylabel('CPU usage (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()

    # Training loss
    plt.plot(hist, label="Training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

mse_acc = mse_acc / len(machine_acc)
print("Average testing RMSE loss: ", mse_acc)

mae_acc = mae_acc / len(machine_acc)
print("Average testing MAE loss: ", mae_acc)

# Input seq. length vs RMSE
# plt.plot(input_acc, mse_res)
# plt.xlabel("Input Sequence Length (time shift)")
# plt.ylabel("RMSE")
# plt.show()
