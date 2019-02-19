import numpy as np
import keras.backend as K

# Load saved data for backtesting
def load_data():
    x = np.load('./data/x.npy')
    y = np.load('./data/y.npy')
    rates = np.load('./data/rates.npy')
    return x, y, rates

# Used in lambda layer
def expand_dimensions(x):
	expX = K.expand_dims(x, axis=1)
	expX = K.expand_dims(expX, axis=1)
	return expX
