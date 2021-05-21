import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from openpyxl import Workbook


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=0.1)
        self.fc_1 = nn.Linear(hidden_size, 16)  # fully connected 1
        self.fc_2 = nn.Linear(16, 8)  # fully connected 1
        self.fc = nn.Linear(8, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 1, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = torch.reshape(x, (1, 1, x.shape[0], x.shape[1]))
        # x = self.cnn_layers(x)
        # x = torch.reshape(x, (x.shape[2], 1, 14))

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn_o = hn.detach().numpy()[-1, :, :]
        hn_o = torch.Tensor(hn_o)
        hn_o = hn_o.view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        hn_1 = hn.detach().numpy()[1, :, :]
        hn_1 = torch.Tensor(hn_1)
        hn_1 = hn_1.view(-1, self.hidden_size)

        # hn_2 = hn.detach().numpy()[3, :, :]
        # hn_2 = torch.Tensor(hn_2)
        # hn_2 = hn_2.view(-1, self.hidden_size)

        out = self.relu(hn_o) + hn_1
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output

        return out


class LSTM2(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size_1, hidden_size_2, nb_layers_1=1, nb_layers_2=1,
                 dropout=0.8):
        super(LSTM2, self).__init__()
        self.num_classes = num_classes  # number of classes
        # self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        # self.seq_length = seq_length  # sequence length

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.nb_layers_1 = nb_layers_1
        self.nb_layers_2 = nb_layers_2
        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size_1, self.nb_layers_1, dropout=dropout)
        self.lstm_2 = nn.LSTM(self.hidden_size_1, self.hidden_size_2, self.nb_layers_2, dropout=dropout)

        self.fc_1 = nn.Linear(hidden_size_2, 8)  # fully connected 1
        self.fc_2 = nn.Linear(8, 8)  # fully connected 1
        self.fc = nn.Linear(8, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_t1 = Variable(torch.zeros(self.nb_layers_1, x.size()[1], self.hidden_size_1))
        c_t1 = Variable(torch.zeros(self.nb_layers_1, x.size()[1], self.hidden_size_1))
        h_t2 = Variable(torch.zeros(self.nb_layers_2, x.size()[1], self.hidden_size_2))
        c_t2 = Variable(torch.zeros(self.nb_layers_2, x.size()[1], self.hidden_size_2))

        h_t1, c_t1 = self.lstm_1(x, (h_t1, c_t1))
        h_t2, _ = self.lstm_2(h_t1, (h_t2, c_t2))
        hn = h_t2.view(-1, self.hidden_size_2)  # reshaping the data for Dense layer next

        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output

        return out


class LSTM3(nn.Module):
    def __init__(self, nb_features=14, hidden_size_1=32, hidden_size_2=64, nb_layers_1=1, nb_layers_2=1,
                 dropout=0.5):  # (self, nb_features=1, hidden_size=100, nb_layers=10, dropout=0.5):
        super(LSTM3, self).__init__()
        self.nb_features = nb_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.nb_layers_1 = nb_layers_1
        self.nb_layers_2 = nb_layers_2

        self.lstm_1 = nn.LSTM(self.nb_features, self.hidden_size_1, self.nb_layers_1, dropout=dropout)
        self.lstm_2 = nn.LSTM(self.hidden_size_1, self.hidden_size_2, self.nb_layers_2, dropout=dropout)

        self.fc_1 = nn.Linear(hidden_size_2, 8)  # fully connected 1
        self.fc_2 = nn.Linear(8, 8)  # fully connected 1
        self.fc = nn.Linear(8, 1)  # fully connected last layer
        self.lin = nn.Linear(self.hidden_size_2, 1)

        self.relu = nn.ReLU()

    def forward(self, input):
        h_t1 = Variable(torch.zeros(self.nb_layers_1, input.size()[1], self.hidden_size_1))
        c_t1 = Variable(torch.zeros(self.nb_layers_1, input.size()[1], self.hidden_size_1))
        h_t2 = Variable(torch.zeros(self.nb_layers_2, input.size()[1], self.hidden_size_2))
        c_t2 = Variable(torch.zeros(self.nb_layers_2, input.size()[1], self.hidden_size_2))

        for i, input_t in enumerate(input.chunk(input.size(1))):
            _, (h_t1, c_t1) = self.lstm_1(input_t, (h_t1, c_t1))
            _, (h_t2, _) = self.lstm_2(h_t1, (h_t2, c_t2))

        hn = h_t2.view(-1, self.hidden_size_2)

        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # first Dense
        out = self.relu(out)  # relu
        outputs = self.fc(out)  # Final Output

        return outputs


def add_remaining_useful_life(df):

    scale = 400  # 225.02895
    shape = 4.40869  # 4.40869
    cut = 140  # 140

    # scale = 214.7489 # 225.02895
    # shape = 4.219719  # 4.40869
    # loc = 10

    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    # remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]

    # Calculate remaining useful life for each row (scaled Weibull)
    min_engine = np.exp(-pow((result_frame["max_cycle"] / scale), shape))
    result_frame["min"] = min_engine

    remaining_useful_life = round(
        (np.exp(-pow((result_frame["time_cycles"] / scale), shape)) - result_frame["min"]) / (
                1 - result_frame["min"]) * cut)

    # Calculate remaining useful life for each row (scaled Weibull)
    # min_engine = np.exp(-pow(((result_frame["max_cycle"] - loc) / scale), shape))
    # result_frame["min"] = min_engine
    #
    # remaining_useful_life = round(
    #     (np.exp(-pow(((result_frame["time_cycles"] - loc) / scale), shape)) - result_frame["min"]) / (
    #             1 - result_frame["min"]) * cut)

    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    result_frame = result_frame.drop("min", axis=1)
    return result_frame


# load data FD001.py
# define filepath to read data
dir_path = './CMAPSSData/'

# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
col_names = index_names + setting_names + sensor_names

# read data
train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

# drop non-informative features, derived from EDA
drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
drop_labels = setting_names + drop_sensors
train.drop(labels=drop_labels, axis=1, inplace=True)

# inspect first few rows
train.head()

title = train.iloc[:, 0:2]
data = train.iloc[:, 2:]

data_norm = (data - data.min()) / (data.max() - data.min())

train_norm = pd.concat([title, data_norm], axis=1)

train_norm = add_remaining_useful_life(train_norm)
# train_norm['RUL'].clip(upper=140, inplace=True)

group = train_norm.groupby(by="unit_nr")

test.drop(labels=drop_labels, axis=1, inplace=True)

title = test.iloc[:, 0:2]
data = test.iloc[:, 2:]

data_norm = (data - data.min()) / (data.max() - data.min())

test_norm = pd.concat([title, data_norm], axis=1)

group_test = test_norm.groupby(by="unit_nr")

num_epochs = 40  # 1000 epochs
learning_rate = 0.01  # 0.001 lr

num_classes = 1  # number of output classes
input_size = 14  # number of features

hidden_size = 64  # number of features in hidden state
num_layers = 4  # number of stacked lstm layers
seq_length = 1

model = LSTM1(num_classes, input_size, hidden_size, num_layers, seq_length)  # our lstm class

# hidden_size_1 = 32  # number of features in hidden state
# hidden_size_2 = 64  # number of features in hidden state

# model = LSTM2(num_classes, input_size, hidden_size_1, hidden_size_2)  # our lstm class

# model = LSTM3()

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 130], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.5)

for epoch in range(num_epochs):

    model.train()
    i = 1

    while i <= 100:
        x = group.get_group(i)
        X = x.iloc[:, 2:-1]
        y = x.iloc[:, -1:]
        X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
        y_train_tensors = Variable(torch.Tensor(y.to_numpy()))
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

        # forward pass
        outputs = model(X_train_tensors_final)
        # outputs = model(X_train_tensors)

        # calculate the gradient, manually setting to 0
        optimizer.zero_grad()

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        # calculates the loss of the loss function
        loss.backward()

        # improve from loss, i.e back propagation
        optimizer.step()

        i += 1

    if epoch % 1 == 0:
        rmse = 0

        j = 1

        result = list()

        # evaluate model
        model.eval()

        while j <= 100:
            x1_test = group_test.get_group(j)
            X_test = x1_test.iloc[:, 2:]

            X_test_tensors = Variable(torch.Tensor(X_test.to_numpy()))

            # reshaping to rows, timestamps, features
            X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

            test_predict = model.forward(X_test_tensors_final)  # forward pass
            # test_predict = model.forward(X_test_tensors)  # forward pass

            data_predict = int(test_predict[-1].detach().numpy())  # numpy conversion

            if data_predict < 0:
                data_predict = 0

            result.append(data_predict)

            rmse += np.power((data_predict - y_test.to_numpy()[j - 1]), 2)

            j += 1

        rmse = np.sqrt(rmse / 100)

        print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, loss.item(), rmse))

    # scheduler.step()

result = y_test.join(pd.DataFrame(result))
result = result.sort_values('RUL', ascending=False)

# the true remaining useful life of the testing samples
true_rul = result.iloc[:, 0:1].to_numpy()
# the predicted remaining useful life of the testing samples
pred_rul = result.iloc[:, 1:].to_numpy()

plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=100, c='r', linestyle='--')  # size of the training set

plt.plot(true_rul, label='Actual Data')  # actual plot
plt.plot(pred_rul, label='Predicted Data')  # predicted plot
plt.title('Remaining Useful Life Prediction')
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Remaining Useful Life")
plt.savefig('L{}N{}({})E{}S{}C{}.png'.format(hidden_size, "16,16", rmse, num_epochs, seq_length, "140"))
# plt.savefig('L({},{})N{}({})E{}S{}.png'.format(hidden_size_1, hidden_size_2, "8,8", rmse, num_epochs, seq_length))
plt.show()
