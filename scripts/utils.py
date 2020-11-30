import numpy as np

from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

def get_data(train_size=50000, val_size=10000, test_size=10000):
  """
  Get the training and testing data for the model.
  """
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.16666, random_state=42)
  x_train = x_train.reshape(train_size, 784)
  x_test = x_test.reshape(test_size, 784)
  x_val = x_val.reshape(val_size, 784)
  x_train = x_train / 255 
  x_test = x_test / 255 
  x_val = x_val / 255
  y_train = utils.to_categorical(y_train, 10)
  y_test = utils.to_categorical(y_test, 10)
  y_val = utils.to_categorical(y_val, 10)

  return x_train, x_test, x_val, y_train, y_test, y_val


def build_model(hp):
    """
    params: hp -- list of all hyperparameters' values, by default:
      hp0 -- 'activation' with possible values ['relu', 'tanh', 'selu']
      hp1 -- 'units_input' with possible values np.linspace(512, 1024, 62)
      hp2 -- 'units_hidden' with possible values np.linspace(128, 512, 64)
      hp3 -- 'learning_rate' with possible values [0.001, 0.005, 0.01]

    Returns the instantiated object of the model.
    """
    model = Sequential()
    activation_choice = hp[0]    
    model.add(Dense(hp[1], input_dim=784, activation=activation_choice))
    model.add(Dense(hp[2], activation=activation_choice))

    ## Using the softmax distribution at the end since we need evaluation
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=Adam(lr=hp[3]), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model
