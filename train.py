from utils import *
#keras stuff
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K

class Portfolio():
    def __init__(self, coins, weights, p_beta, k, learning_rate, mini_batch_count, mini_batch_size, epochs):
        self.coins = coins
        self.weights = weights
        self.p_beta = p_beta
        self.k = k
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.mini_batch_count = mini_batch_count
        self.epochs = epochs

    def create_eiie(self, input_tensor_shape, rates_shape)
        K.set_image_data_format("channels_first")
        #3 inputs: main input, weight input, bias input
        main_input = Sequential()
        main_input.add(Conv2D(2, kernal_size=(3, 1), activation='relu',  input_shape=input_tensor_shape))
        main_input.add(Conv2D(20, kernal_size=(48, 1), activation='relu'))




if __name__ == "__main__":
    #load saved data
    x, y, rates = load_data()
    #init params for portfolio
    coins = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'ICX/BTC', 'XRP/BTC', 'XLM/BTC', 'NEO/BTC', 'LTC/BTC', 'ADA/BTC']
    initial_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p_beta = 0.00005
    k = 15
    learning_rate = 0.00019
    mini_batch_count = 30
    mini_batch_size = 50
    epochs = 50

    portfolio = Portfolio(coins, initial_weights, p_beta, k, learning_rate, mini_batch_count, mini_batch_size, epochs)
    portfolio.create_eiie(x.shape[1:], rates.shape[1:])
