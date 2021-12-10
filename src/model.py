import tensorflow as tf
from tensorflow.keras import datasets, preprocessing, models, layers
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from config import cfg


def NN(input_dim: tuple, num_output: int):
    model = models.Sequential([
        layers.Dense(100, input_shape=input_dim, activation='relu'),
        layers.Flatten(),
        layers.Dense(num_output)
    ])
    return model


def RNN(input_dim: tuple, num_output: int):
    model = models.Sequential([
        layers.SimpleRNN(100, input_shape=input_dim,
                         dropout=0.5, recurrent_dropout=0.5),
        layers.Dense(num_output)
    ])

    return model


def LSTM(input_dim: tuple, num_output: int):
    model = models.Sequential([
        layers.Bidirectional(layers.LSTM(
            100, return_sequences=True, recurrent_dropout=0.5), input_shape=input_dim),
        layers.Bidirectional(layers.LSTM(100, recurrent_dropout=0.5)),
        layers.Dense(num_output)
    ])

    return model


def Model(input_dim: tuple, num_output=int):
    opt = cfg
    if opt.model == 'NN':
        return NN(input_dim, num_output)
    elif opt.model == 'RNN':
        return RNN(input_dim, num_output)
    elif opt.model == 'LSTM':
        return LSTM(input_dim, num_output)
    else:
        return print("Please enter a valid Model")
