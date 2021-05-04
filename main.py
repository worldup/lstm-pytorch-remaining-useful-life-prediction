import torch
import torch.nn as nn
from torch.autograd import Variable

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
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
drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
drop_labels = setting_names + drop_sensors
train.drop(labels=drop_labels, axis=1, inplace=True)

train = add_remaining_useful_life(train)
# train[index_names + ['RUL']].head()

# clip RUL as discussed in SVR and problem framing analysis
train['RUL'].clip(upper=130, inplace=True)

# inspect first few rows
train.head()

title = train.iloc[:, 0:2]
data = train.iloc[:, 2:]

# mm = MinMaxScaler()
# ss = StandardScaler()
# normalized_df=(df-df.mean())/df.std()

data_norm = (data - data.min()) / (data.max() - data.min())

train_norm = pd.concat([title, data_norm], axis=1)

group = train_norm.groupby(by="unit_nr")

x1 = group.get_group(1)

X = x1.iloc[:, 2:-1]
y = x1.iloc[:, -1:]

X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
y_train_tensors = Variable(torch.Tensor(y.to_numpy()))

# reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))
# X_train_tensors_final.shape
# y_train_tensors.shape

num_epochs = 1000  # 1000 epochs
learning_rate = 0.001  # 0.001 lr

input_size = 15  # number of features
hidden_size = 64  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

num_classes = 1  # number of output classes

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])  # our lstm class

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    i = 1
    while i <= 100:
        x = group.get_group(i)
        X = x.iloc[:, 2:-1]
        y = x.iloc[:, -1:]
        X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
        y_train_tensors = Variable(torch.Tensor(y.to_numpy()))
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

        outputs = lstm1.forward(X_train_tensors_final)  # forward pass
        optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

        # obtain the loss function
        loss = criterion(outputs, y_train_tensors)

        loss.backward()  # calculates the loss of the loss function

        optimizer.step()  # improve from loss, i.e backprop

        i += 1

    if epoch % 10 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



test.drop(labels=drop_labels, axis=1, inplace=True)

title = test.iloc[:, 0:2]
data = test.iloc[:, 2:]

test_norm = (data - data.min()) / (data.max() - data.min())

test_norm = pd.concat([title, data_norm], axis=1)

group = test_norm.groupby(by="unit_nr")

rmse = 0

j = 1

while j <= 100:
    x1 = group.get_group(1)
    X = x1.iloc[:, 2:-1]

    X_test_tensors = Variable(torch.Tensor(X.to_numpy()))

    # reshaping to rows, timestamps, features
    X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

    test_predict = lstm1(X_test_tensors_final) # forward pass
    data_predict = test_predict.data.numpy()[-1] * 130 # numpy conversion

    rmse += np.power((data_predict - y_test.to_numpy()[j-1]), 2)

    j += 1

rmse = np.sqrt(rmse/100)
print(rmse)