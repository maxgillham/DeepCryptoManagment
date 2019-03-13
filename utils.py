import numpy as np
import keras.backend as K
import pickle

# Load saved data for backtesting
def load_data():
    # Open with pickling
    with open('./data/x.txt', 'rb') as fp: x = pickle.load(fp)
    with open('./data/y.txt', 'rb') as fp: y = pickle.load(fp)
    with open('./data/rates.txt', 'rb') as fp: rates = pickle.load(fp)
    return x, y, rates

# Used in lambda layer
def expand_dimensions(x):
	expX = K.expand_dims(x, axis=1)
	expX = K.expand_dims(expX, axis=1)
	return expX
