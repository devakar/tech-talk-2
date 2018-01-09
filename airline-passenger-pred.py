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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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

log_dir = "2-layer-4-4-LeakyReLU"

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
logger = TensorBoard(log_dir='log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
csv_logger = CSVLogger('training-2-layer-32-32-LeakyReLU-noScale-noDiff.csv')

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


# difference the data to make it stationary for better fit
diff_values = difference(values, 1)
# diff_values = values


# scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(diff_values)


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

# load model
model = load_model('airline-model-2-layer-4-4-LeakyReLU.h5')

# predictions for testing data
yhat= model.predict(test_X, batch_size=1)
# reshape output to 2D array
yhat = yhat.reshape(yhat.shape[0], yhat.shape[2])
# inverse scaling
yhat_inv = scaler.inverse_transform(yhat)
print("shape of output after inverse scaling", yhat.shape)
#inverse differencing
yhat_final = inverse_diff(yhat_inv, values, n_train_months)

# make array of expectated and predicted values
ex_start = n_train_months+1
ex_end = n_train_months+1+yhat_final.shape[0]
expectation = np.array(values[ex_start:ex_end,0:1])
predictions = np.array(yhat_final[:,0:])

mem = np.concatenate((expectation,predictions), axis=1)
np.savetxt("expec-pred-airline-model-2-layer-4-4-LeakyReLU.csv", mem, delimiter=",")

rmse = sqrt(mean_squared_error(expectation, predictions))
print('Test RMSE: %.3f' % rmse)

plt.plot(expectation, label='Actual')
plt.plot(predictions, label= 'Predicted')
plt.legend(loc='upper right')
plt.ylabel('Numbner of passenger')
plt.xlabel('Time Instant')
plt.title("Prediction of airline passenger used from LSTM Network")
plt.savefig('airline-model-2-layer-4-4-LeakyReLU.png')
plt.show()

minimum_exp = np.amin(expectation)
maximum_exp = np.amax(expectation)
mean_exp = np.mean(expectation)

print("max of expectation", maximum_exp)
print("min of expectation", minimum_exp)
print("mean of expectation", mean_exp)

diff_max_min = maximum_exp -minimum_exp

print("Normalised RMSE:  ", rmse/diff_max_min)
print("Coefficient of Variance ", rmse/mean_exp)

mae = mean_absolute_error(expectation, predictions)
print("TEST MAE: %.3f" %mae)

print("Test mean_absolute_percentage_error", np.mean(np.abs((expectation - predictions) / expectation)) * 100)

r_square = r2_score(expectation, predictions)
print("TEST R_sqaure: %.3f" %r_square)
