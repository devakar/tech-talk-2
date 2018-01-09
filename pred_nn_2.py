import numpy as np
import matplotlib.pylab as plt

from keras.models import load_model


dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
print("shape of original dataset", dataset.shape)

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8:]
print("shape of input and ouput dataset", X.shape, Y.shape)

model = load_model('model-2-layer-12-8-relu-relu-150Epochs.h5')

# calculate predictions
predictions = model.predict(X)
rounded = np.ndarray.round(predictions)
expec_pred = np.concatenate((Y,rounded), axis=1)
accuracy = (Y == rounded).mean()
print("accuracy of the model", accuracy)
np.savetxt("expec-pred-model-2-layer-12-8-relu-relu-150Epochs.csv", expec_pred, delimiter=",")

