# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
# packages and libraries

# TODO:
#
# PYTHON: finish LSTM
#
# SCALA:
# - create new variable mid_price - (maybe some transformation - normalization / log / ...) - OK
# - change date variable to actual date - DROPPED THE IDEA
# - find out the data that is working
# - sort data - OK

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
import numpy as np


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def read_data(file_loc, ratio=None, validation_size=None):
    """
    :param file_loc: path of the file in the computer
    :param ratio: ratio between training and testing data
    :param validation_size: how many samples will be used for the validation dataset
    :return: a dictionary of datasets with the complete one, training, testing and validation
    """
    data = pd.DataFrame(pd.read_csv(file_loc, header=None,
                                    names=[
                                        'conversionSymbol',
                                        'volumeto',
                                        'high',
                                        'low',
                                        'time',
                                        'volumefrom',
                                        'close',
                                        'open',
                                        'conversionType',
                                        'midPrice',
                                        'midPriceNorm',
                                        'empty'
                                    ]))

    print(data.head())
    print(data.dtypes)
    data = data.drop(
        ["conversionSymbol", "volumeto", "low", "close", "open", "conversionType", "midPrice", "high", "empty"], axis=1)

    if ratio is not None:
        training_ratio = int(ratio * data.shape[0])
        training_data = data[:training_ratio]
        testing_data = data[training_ratio:data.shape[0] - validation_size]
        validation_data = data[data.shape[0] - validation_size:]
        return {'complete': data, 'train': training_data, 'test': testing_data, 'validation': validation_data}

    else:
        return {'complete': data}


def plot_time_series(data, time_range, variable):
    plt.figure(figsize=(18, 9))
    if len(variable) == 2:
        plt.plot(range(data.shape[0]), (data[variable[0]] + data[variable[1]]) / 2.0)
        y_label = 'Mid price'
    if len(variable) == 1:
        plt.plot(range(data.shape[0]), (data[variable[0]]) / 2.0)
        y_label = variable[0]
    plt.xticks(range(0, data.shape[0], time_range), data['time'].loc[::time_range], rotation=45)
    plt.xlabel('time', fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.show()


def create_lstm_model(data):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1:])))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=["accuracy"])
    return model


def main():
    df = read_data("../PreProcessing/data_cleaned/part-00000", ratio=2 / 3, validation_size=10)
    # plot_time_series(data=df['complete'], time_range=500, variable=['high'])
    x_train, y_train = data_labeling(data=df["train"])
    model = create_lstm_model(data=x_train)
    model.fit(x_train, y_train, epochs=500, batch_size=32)


def data_labeling(data):
    x_data = data[:-1].to_numpy()
    y_data = data[1:]
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data["midPriceNorm"].to_numpy()


if __name__ == "__main__":
    main()
