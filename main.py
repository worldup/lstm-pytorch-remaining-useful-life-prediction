import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    min_engine = np.exp(-pow((max_cycle / 225.03), 4.41))
    max_engine = np.exp(-pow((1 / 225.03), 4.41))

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row
    # remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    remaining_useful_life = round(130 * ((np.exp(-pow((result_frame["time_cycles"] / 225.03), 4.41))-min_engine)/(max_engine-min_engine)))
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

# inspect first few rows
train.head()

# train = add_remaining_useful_life(train)
# train[index_names + ['RUL']].head()

# clip RUL as discussed in SVR and problem framing analysis
# train['RUL'].clip(upper=125, inplace=True)

# drop non-informative features, derived from EDA
drop_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
drop_labels = setting_names + drop_sensors
train.drop(labels=drop_labels, axis=1, inplace=True)

remaining_sensors = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9',
                     's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']

# train['breakdown'] = 0
# idx_last_record = train.reset_index().groupby(by='unit_nr')['index'].last()  # engines breakdown at the last cycle
# train.at[idx_last_record, 'breakdown'] = 1

# train['start'] = train['time_cycles'] - 1
# train.tail()  # check results

# cut_off = 200
# train_censored = train[train['time_cycles'] <= cut_off].copy()

# data = train_censored[index_names + ['breakdown']].groupby('unit_nr').last()

X = train.iloc[:, 2:]
# y = df.iloc[:, 5:6]

mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
# y_mm = mm.fit_transform(y)