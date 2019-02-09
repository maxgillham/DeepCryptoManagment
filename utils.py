import numpy as np

def load_data():
    x = np.load('./data/x.npy')
    y = np.load('./data/y.npy')
    rates = np.load('./data/rates.npy')
    return x, y, rates

if __name__ == "__main__":
    x, y, rates = load_data()
    print(x.shape)
    print(y.shape)
    print(rates.shape)

    print(x[0])
    print(y[0])
    print(rates[0])
