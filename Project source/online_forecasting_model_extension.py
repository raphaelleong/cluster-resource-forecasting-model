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
train_size = 1
# removed batching
batch_size = train_size
valid_size = 24
test_size = 0

# tau
input_size = 288

# size of hidden layers
learning_rate = 0.004
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
        # self.channels = int(batch_size / input_dim)
        self.channels = 1

        # Conv1d with lstm
        self.conv1 = nn.Conv1d(in_channels=self.channels, out_channels=self.channels,
                               kernel_size=4, padding=1)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.lstm_x = nn.LSTM(self.input_dim, 128, 2)
        self.lstm_y = nn.LSTM(128, 64, 2)

        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        # ignoring hidden states for it to be stateful
        # CNN LSTM
        conv_out = self.conv1(x)
        max_out = self.pooling(conv_out)
        lstm_out, self.hidden = self.lstm_x(max_out)
        lstm_out, self.hidden = self.lstm_y(lstm_out)
        out = self.linear(lstm_out)

        return out.view(-1)


# machine_acc = [348805021, 350588109, 4820285492, 1436333635,
#                3338000908, 1390835522, 1391018274, 5015788232, 4874102959]
machine_acc = [350588109]

mse_acc = 0
mse_res = []
mae_acc = 0

# Online training and forecasting
for m_id in machine_acc:
    # Data pre processing
    data_train, data_test, _ = load_data_from_csv(m_id, train_size, valid_size, test_size,
                                                  input_size + 1, faster_sampling)

    data_train = torch.from_numpy(data_train).type(torch.Tensor)

    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size)

    model = LSTM(input_size, batch_size=batch_size)
    loss_fn = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training
    hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        for batch_idx, batch_sample in enumerate(dataloader_train):
            bs = batch_sample[:, :input_size]
            X_train = bs.view([train_size, -1, input_size])
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
    print("================================================")
    print(y_pred)
    print(y_target)
    # Testing evaluation

    # model.eval()
    cache = []
    cache_target = []
    for temp in range(1, valid_size + 1):
        data_eval = data_test[temp - 1:temp]

        ch_index = 0
        for c in cache:
            length = len(data_eval[0])
            data_eval[0][length - temp + ch_index] = c
            ch_index += 1
        data_eval = torch.from_numpy(data_eval).type(torch.Tensor)

        bs = data_eval[:, :input_size]

        train = bs.view([1, -1, input_size])
        target = data_eval[:, input_size:]

        pred = model(train)
        loss = loss_fn(pred, target)
        print("Predicted: ", pred.detach().numpy()[0], "%")
        print("Target: ", target.detach().numpy()[0][0], "%")
        cache.append(pred.detach().numpy()[0])
        cache_target.append(target.detach().numpy()[0][0])

        test_rmse = np.sqrt(loss_fn(pred, target).item())
        print("Machine ID: ", m_id, "RMSE: ", test_rmse)
        print("================================================")
    plot_size = valid_size

    x_time = []
    for s in range(0, plot_size * 5, 5):
        x_time.append(s)

    # Evaluation

    # Predictions vs Target
    plt.plot(x_time, cache, label="Predictions")
    plt.plot(x_time, cache_target, label="Data")
    plt.xlabel('Time (min)')
    plt.ylabel('CPU usage (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.show()
    #
    # Training loss
    plt.plot(hist, label="Training loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
