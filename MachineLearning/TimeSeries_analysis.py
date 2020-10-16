# https://www.datacamp.com/community/tutorials/lstm-python-stock-market
# packages and libraries

# TODO:
#
# PYTHON: finish LSTM
# Use 72 TimeStamps in the lstm to predict future value - OK
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
from tqdm import tqdm
import random


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def read_data(file_loc, features, predicted_feature, ratio=None, testing_size=None):
    """
    :param features: parameters that are going to be used to fit the model
    :param file_loc: path of the file in the computer
    :param ratio: ratio between training and testing data
    :param testing_size: how many samples will be used for the testing dataset
    :return: a dictionary of datasets with the complete one, training, testing and validation
    """
    data = pd.DataFrame(pd.read_csv(file_loc, sep=',', index_col=False,
                                    names=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close',
                                           'converstionType', 'conversionSymbol', 'midPrice', 'midPriceNorm',
                                           'volumefromnorm', 'volumetonorm', 'empty']))

    print(data.head())
    print(data.dtypes)

    data["time"] = data["time"].str[1:].astype(float)

    data.set_index("time", inplace=True)
    features.append(predicted_feature)
    data = data[features]

    print(data.head())
    print(data.dtypes)

    data['value_to_predict'] = data[predicted_feature].shift(-1)
    # data['classification_to_predict'] = list(map(classify, data['close'], data['future']))

    training_data, validation_data, testing_data = splitting_datasets(data, ratio, testing_size)

    return {'complete': data, 'train': training_data, 'validation': validation_data, 'test': testing_data}


def splitting_datasets(data, ratio, testing_size):
    if ratio is None:
        print("by default ratio is set to 0.66")
        training_ratio = int(0.66 * data.shape[0])
    else:
        training_ratio = int(ratio * data.shape[0])
    training_data = data[:training_ratio]
    if testing_size is None:
        print("by default testing size is set to 10 samples")
        validation_data = data[training_ratio:data.shape[0] - 10]
        testing_data = data[data.shape[0] - 10:]
    else:
        validation_data = data[training_ratio:data.shape[0] - testing_size]
        testing_data = data[data.shape[0] - testing_size:]

    return training_data, validation_data, testing_data


def createTimeWindows(data, timestamp_size):
    """
    We should consider the the previous data when predicting. In this case with a window_length size
    :param timestamp_size: How many previous data to consider
    :param data: training data
    :return: (nr_data_samples-window_length, window_length, nr_features)
    """
    #
    shape = (data.shape[0] - timestamp_size + 1, timestamp_size, data.shape[1])
    t_minus_window_data = np.empty(shape)
    timestamp_data = deque(maxlen=timestamp_size)

    counter = 0
    for i in data.values:  # iterate over the values
        timestamp_data.append([n for n in i])  # store all but the target
        if len(timestamp_data) == timestamp_size:
            t_minus_window_data[counter] = np.array(timestamp_data)
            counter += 1

    np.random.shuffle(t_minus_window_data)  # shuffle for good measure.

    # todo: add data_labelling to get labels also
    # IT WORKS, ALTHOUGH IT IS SUPID

    # shape = (data.shape[0], window_length, data.shape[1])
    # t_minus_window_data = np.empty(shape)
    # for t_i, i in tqdm(enumerate(data.values)):
    #     window_counter = 0
    #     for t_j, j in enumerate(data.values):
    #         if t_j >= t_i: break
    #         if t_i - t_j <= window_length and t_i >= window_length:
    #             t_minus_window_data[t_i][window_counter] = j
    #             window_counter += 1

    return t_minus_window_data


def data_labeling(data, feature, timestamp_size):
    x_data = createTimeWindows(data[:-1], timestamp_size=timestamp_size)
    # x_data = data[:-1]#.to_numpy()
    y_data = data[timestamp_size:]
    # x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
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
    features = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'midPrice', 'volumefromnorm', 'volumetonorm']
    predicted_feature = 'close'
    timestamp_size = 72
    df = read_data("../PreProcessing/data_cleaned/part-00000", features=features, predicted_feature=predicted_feature,
                   ratio=3 / 5, testing_size=10)
    # plot_time_series(data=df['complete'], time_range=500, variable=['midPriceNorm'])
    x_train, y_train = data_labeling(data=df["train"], features=features, timestamp_size=timestamp_size)
    validation_data = data_labeling(data=df["validation"], features=features, timestamp_size=timestamp_size)
    model = create_lstm_model(data=x_train)
    training_history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=validation_data)
    plot_loss(training_history)
    # x_test, _ = data_labeling(data=df["test"], feature=feature)
    # y = model.predict(df["test"][feature])


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
