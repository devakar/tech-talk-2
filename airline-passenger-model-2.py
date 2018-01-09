import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas import DataFrame
from pandas import read_csv
from pandas import concat

from numpy import log
from math import sqrt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import adfuller

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import Dropout
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

log_dir = "airline-model-2-layer-4-4-LeakyReLU"

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
logger = TensorBoard(log_dir='airline-log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
csv_logger = CSVLogger('training-airline-model-2-layer-4-4-LeakyReLU.csv')

# create a differenced series
def difference(dataset, interval=1):
    print(dataset.ndim)
    print(dataset.shape)
    diff = np.ndarray(shape=(dataset.shape[0]-interval, dataset.shape[1]), dtype=float)
    for i in range(interval, len(dataset)):
        for j in range(dataset.shape[1]):
            diff[i-1][j] = dataset[i][j] - dataset[i - interval][j]
    return diff

def inverse_diff(y, dataset, split):
    print(y.shape, dataset.shape, split)
    l = split+y.shape[0]
    x = dataset[split:l,]
    return y+x

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg    

# load data 
df = read_csv('international-airline-passengers.csv', header=0, index_col=0)
df.dropna(inplace=True)
print(df.head())
values = df.values

# check data is in float
values = values.astype('float32')

# plot data
plt.plot(values)
plt.show()
# histogram of data
plt.hist(values)
plt.show()

# difference the data to make it stationary for better fit
diff_values = difference(values, 1)
# diff_values = values

# plot differenciated data
plt.plot(diff_values)
plt.show()
# histogram of differenciated data
plt.hist(diff_values)
plt.show()

# scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(diff_values)

# plot scaled data
plt.plot(scaled)
plt.show()
# histogram of differenciated data
plt.hist(scaled)
plt.show()

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
print(reframed.head())

# split into train and test sets
reframed_values = reframed.values
print("shape of reframed values", reframed_values.shape)

n_train_months = 96
train = reframed_values[:n_train_months, :]
test = reframed_values[n_train_months:, :]

# split into input and outputs
train_X, train_y = train[:, :1], train[:, 1:]
test_X, test_y = test[:, :1], test[:, 1:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
train_y = train_y.reshape((train_y.shape[0], 1, train_y.shape[1]))
test_y = test_y.reshape((test_y.shape[0], 1, test_y.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), return_sequences=True, name='hidden-layer-1'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(LSTM(4, return_sequences=True, name='hidden-layer-2'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
# model.add(LSTM(32, return_sequences=True, name='hidden-layer-3'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=.001))
model.add(Dense(1, name='output-layer'))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=1, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[logger, reduce_lr, csv_logger])

model.save('airline-model-2-layer-4-4-LeakyReLU.h5')
print(model.summary())

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(loc='upper right')
plt.savefig('train-test-error-airline-model-2-layer-4-4-LeakyReLU.png')


