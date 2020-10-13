# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
# packages and libraries

# TODO:
#
# PYTHON: finish LSTM
# Use 72 TimeStamps in the lstm to predict future value
# Look over model, we can probably add more (batchNorm, more units, layers, ...)
# SCALA:
# Clean raw API response in Scala (once everything works)
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
import random

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def read_data(file_loc, feature, ratio=None, validation_size=None):
    """
    :param file_loc: path of the file in the computer
    :param ratio: ratio between training and testing data
    :param validation_size: how many samples will be used for the validation dataset
    :return: a dictionary of datasets with the complete one, training, testing and validation
    """
    data = pd.DataFrame(pd.read_csv(file_loc, header=None,
                                    names=[
                                        'time',
                                        'high',
                                        'low',
                                        'open',
                                        'close',
                                        'volumefrom',
                                        'volumeto',
                                        'conversionType',
                                        'conversionSymbol',
                                        'midPrice',
                                        'midPriceNorm',
                                        'volumefromNorm',
                                        'volumetoNorm',
                                        'empty'
                                    ]))

    print(data.head())
    print(data.dtypes)

    # data = data.drop(
    #     ["conversionSymbol", "volumeto", "low", "close", "open", "conversionType", "midPrice", "empty", feature], axis=1)

    data = data[['time', 'volumefromNorm', 'volumetoNorm',  feature]]
    data['time'] = data['time'].str[1:].astype(float)

    print(data.head())
    print(data.dtypes)

    window_length = 72
    if ratio is not None:
        training_ratio = int(ratio * data.shape[0])
        training_data = data[:training_ratio]
        # training_data = createTimeWindows(data[:training_ratio], window_length=window_length)
        testing_data = data[training_ratio:data.shape[0] - validation_size]
        validation_data = data[data.shape[0] - validation_size:]
        return {'complete': data, 'train': training_data, 'test': testing_data, 'validation': validation_data}

    else:
        return {'complete': data}

def createTimeWindows(data, window_length=72):
    """
    We should consider the the previous data when predicting. In this case with a window_length size
    :param window_length: How many previous data to consider
    :param data:
    :return: (nr_data_samples-window_length, window_length, nr_features)
    """
    #

    # sequential_data = []
    # prev_days = deque(maxlen=window_length)
    #
    # for i in data.values:  # iterate over the values
    #     prev_days.append([n for n in i[:-1]])  # store all but the target
    #     if len(prev_days) == window_length:
    #         sequential_data.append([np.array(prev_days), i[-1]])
    #
    # random.shuffle(sequential_data)  # shuffle for good measure.

    # todo add data_labelling to get labels also

def data_labeling(data, feature):
    x_data = data[:-1].to_numpy()
    y_data = data[1:]
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    return x_data, y_data[feature].to_numpy()

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
    # model = Sequential()
    # model.add(LSTM(128, input_shape=(data.shape[1:]), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.
    #
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.1))
    # model.add(BatchNormalization())
    #
    # model.add(LSTM(128))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    #
    # model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.2))
    # opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    # model.compile(optimizer=opt, loss='mean_squared_error', metrics=["accuracy"])
    # return model



def main():
    feature = "midPriceNorm"
    df = read_data("../PreProcessing/data_cleaned/part-00000", feature=feature, ratio=2 / 3, validation_size=10)
    # plot_time_series(data=df['complete'], time_range=500, variable=['midPriceNorm'])
    x_train, y_train = data_labeling(data=df["train"], feature=feature)
    validation_data = data_labeling(data=df["validation"], feature=feature)
    model = create_lstm_model(data=x_train)
    training_history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=validation_data)
    plot_loss(training_history)
    x_test, _ = data_labeling(data=df["test"], feature=feature)
    y = model.predict(x_test)





def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
