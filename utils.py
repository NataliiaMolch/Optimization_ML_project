import numpy as np


import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations

from sklearn.model_selection import train_test_split


activation = {
    1. : 'relu', 2. : 'selu', 3. : 'tanh'
}
hp_dictionary = {
    'activation': ['relu', 'selu', 'tanh'],
    'beta_one': [0.7, 0.8, 0.9],
    'beta_two': [0.9, 0.95, 0.999],
    'learning_rate': [0.001, 0.005, 0.1]
}

def get_data(train_size=50000, val_size=10000, test_size=10000):
  """
  Get the training and testing data for the model.
  """
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.16666, random_state=42)
  x_train = x_train.reshape(train_size, 784)
  x_test = x_test.reshape(test_size, 784)
  x_val = x_val.reshape(val_size, 784)
  x_train = x_train / 255 
  x_test = x_test / 255 
  x_val = x_val / 255
  y_train = utils.to_categorical(y_train, 10)
  y_test = utils.to_categorical(y_test, 10)
  y_val = utils.to_categorical(y_val, 10)

  return x_train, np.vstack([x_test, x_val]), y_train, np.vstack([y_test, y_val])


def model(hp = [activations.relu, 0.9, 0.999, 1e-3]):
    """Returning the instance of the model"""
    model = Sequential()
    activation_choice = hp[0]
    model.add(Dense(
        512, input_dim=784, activation=activation_choice))
    model.add(Dense(
        256, activation=activation_choice))   
    model.add(Dense(10, activation='softmax'))
    model.compile(
        optimizer=Adam(lr=hp[3], beta_1=hp[1], beta_2=hp[2]),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])
    return model
