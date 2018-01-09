import numpy as np
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger

log_dir = "2-layer-12-8-relu-relu-150Epochs"

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
logger = TensorBoard(log_dir='rnn-log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
csv_logger = CSVLogger('2-layer-12-8-relu-relu-150Epochs.csv')

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
print("shape of original dataset", dataset.shape)

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8:]
print("shape of input and ouput dataset", X.shape, Y.shape)

split1 = int(len(X)*0.6)
split2 = int(len(X)*0.2)
split3 = split1 + split2 

# split dataset into training, validation and test dataset
# dataset for training
train_x = X[:split1,:]
train_y = Y[:split1,:]
print("shape of training data", train_x.shape, train_y.shape)

# dataset for validation
valid_x = X[split1:split3,:]
valid_y = Y[split1:split3,:]
print("shape of validation data", valid_x.shape, valid_y.shape)

# dataset for testing
test_x = X[split3:,:]
test_y = Y[split3:,:]
print("shape of testing data", test_x.shape, test_y.shape)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', name='hidden-layer-1'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=.001))
model.add(Dense(8, activation='relu', name='hidden-layer-2'))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())
# model.add(LeakyReLU(alpha=.001))
model.add(Dense(1, activation='sigmoid', name='output-layer'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(train_x, train_y, epochs=150, batch_size=1, validation_data=(valid_x, valid_y),  verbose=2, shuffle=True, callbacks=[logger, reduce_lr, csv_logger, early_stopping])

# visualize the train and validation error 
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend(loc='upper right')
plt.savefig('train-test-error-model-2-layer-12-8-relu-relu-150Epochs.png')
plt.show()
# evaluate the model
scores = model.evaluate(test_x, test_y)
print("Score on the test data \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(X)
print("shape of predictions", predictions.shape)
# round predictions
rounded = np.ndarray.round(predictions)
expec_pred = np.concatenate((Y,rounded), axis=1)
accuracy = (Y == rounded).mean()
print("accuracy of the model", accuracy*100)
np.savetxt("expec-pred-model-2-layer-12-8-relu-relu-150Epochs.csv", expec_pred, delimiter=",")


print(model.summary())
model.save('model-2-layer-12-8-relu-relu-150Epochs.h5')
del model



