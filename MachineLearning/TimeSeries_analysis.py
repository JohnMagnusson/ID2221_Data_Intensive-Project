# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
# packages and libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM


def read_data(file_loc, ratio=None, validation_size=None):
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
                                        'conversionType'
                                    ]))

    print(data.head())
    print(data.dtypes)

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


def lstm_prediction(data):
    model = Sequential()
    model.add(LSTM)
    return 1


def main():
    df = read_data("../PreProcessing/data_cleaned/part-00000", ratio=2 / 3, validation_size=10)
    plot_time_series(data=df['complete'], time_range=500, variable=['high'])


if __name__ == "__main__":
    main()
