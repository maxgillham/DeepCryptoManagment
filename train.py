from utils import *
import tensorflow as tf
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

    def create_eiie(self, input_tensor_shape, rates_shape):
        K.set_image_data_format("channels_first")
        # 3 inputs: main input, weight input, bias input
        main_input = Input(shape=input_tensor_shape, name='MainInput')
        weight_input = Input(shape=rates_shape, name='WeightInput')
        bias_input = Input(shape=(1, ), name='BiasInput')

        # Assemble layers using the Functional API
        # Starting with main input
        x = Conv2D(2, kernel_size=(3, 1), activation='relu')(main_input)
        x = Conv2D(20, kernel_size=(48, 1), activation='relu')(x)
        # Adding the weight input. Note - A Lambda layer wraps a expression as a Layer object
        # Here we call the expand_dimensions method defined in utils.py
        weight_input_expression = Lambda(expand_dimensions)(weight_input)
        x = Concatenate(axis=1)([x, weight_input_expression])
        x = Conv2D(1, kernel_size=(1, 1))(x)
        # Adding the bias input. Note - Same Lambda layer with expand_dimensions methods in utils.py
        bias_input_expression = Lambda(expand_dimensions)(bias_input)
        x = Concatenate(axis=3)([x, bias_input_expression])
        model_output = Activation('softmax')(x)

        # Inputs and output to create the ensemble
        self.model = Model([main_input, weight_input, bias_input], model_output)

        # Creating the custom symbolbolic gradiant
        mu = K.placeholder(shape=(None, 1), name='mu')
        y = K.placeholder(shape=(None, len(self.coins)), name='y')

        sqOut = K.squeeze(K.squeeze(self.model.output, 1), 1)
        yOutMult = tf.multiply(sqOut, y)
        yOutBatchDot = tf.reduce_sum(yOutMult, axis=1, keep_dims=True)
        muDotMult = tf.multiply(mu, yOutBatchDot)

        loss = -K.log(muDotMult)

        grad = K.gradients(loss, self.model.trainable_weights)
        self.compute_gradient = K.function(inputs=[main_input, weight_input, bias_input, mu, y, self.model.output], outputs=grad)
        self.model.summary()
        return


if __name__ == "__main__":
    #load saved data
    x, y, rates = load_data()
    # init params for portfolio
    coins = ['EOS/BTC', 'ETH/BTC', 'ETC/BTC', 'TRX/BTC', 'ICX/BTC', 'XRP/BTC', 'XLM/BTC', 'NEO/BTC', 'LTC/BTC', 'ADA/BTC']
    initial_weights = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    p_beta = 0.00005
    k = 15
    learning_rate = 0.00019
    mini_batch_count = 30
    mini_batch_size = 50
    epochs = 50

    portfolio = Portfolio(coins, initial_weights, p_beta, k, learning_rate, mini_batch_count, mini_batch_size, epochs)
    portfolio.create_eiie(np.array(x).shape[1:], np.array(y).shape[1:])
