from pathlib import Path
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from config import cfg
from model import Model
from util.build_feature import preprocess
from util.plot import plotting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


opt = cfg
data = preprocess(opt.data_Id)


def featureLabelSplit(df, target=opt.target) -> tuple:
    Y = df[target].to_numpy()
    X = df.to_numpy()
    return (X, Y)


def data_split(X, Y, test_ratio=0.3):
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_ratio, shuffle=False)
    return X_train, Y_train, X_val, Y_val


X, Y = featureLabelSplit(data)
print(type(X))
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X_train, Y_train, X_val, Y_val = data_split(X, Y)

print(type(X_train))

X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train)
X_val = scaler_X.transform(X_val)
Y_val = scaler_Y.transform(Y_val)

print(type(X_train))

batch_train = TimeseriesGenerator(data=X_train, targets=Y_train.reshape(
    -1, 1), length=24, sampling_rate=1, batch_size=opt.batch_size)

batch_val = TimeseriesGenerator(data=X_val, targets=Y_val.reshape(
    -1, 1), length=24, sampling_rate=1, batch_size=opt.test_batch_size)

model = Model(input_dim=batch_train[0]
              [0].shape[1:], num_output=len(opt.target))

path = '../models'

if not os.path.exists(path):
    os.makedirs(path)
checkpoint_path = path + f'/{opt.data_Id}_{opt.model}_model.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer='adam', loss='mse')
history = model.fit(batch_train, epochs=opt.epochs,
                    validation_data=batch_val, validation_freq=1, workers=2, callbacks=[cp_callback])

plotting(history=history.history)
