# packages and libraries
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
import os


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def read_data(folder_loc, features, predicted_feature):
    """
    :param predicted_feature: parameter that is going to be predicted
    :param features: parameters that are going to be used to fit the model
    :param folder_loc: path of the folder with all 3 data in the computer
    :return: a dictionary of datasets with the complete one, training, testing and validation
    """

    data = {}
    features.append(predicted_feature)
    for datasets_folder in os.listdir(folder_loc):
        for file in os.listdir(f'{folder_loc}/{datasets_folder}'):
            if file == "part-00000":
                data_loc = f'{folder_loc}/{datasets_folder}/{file}'
                data_temp = pd.DataFrame(pd.read_csv(data_loc, sep=',', index_col=False,
                                                     names=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto',
                                                            'close', 'converstionType', 'conversionSymbol', 'midPrice',
                                                            'midPriceNorm', 'volumefromNorm', 'volumetoNorm', 'empty']))

                data_temp["time"] = data_temp["time"].str[1:].astype(float)

                data_temp.set_index("time", inplace=True)

                if features is not None:
                    data_temp = data_temp[features]

                data_temp['value_to_predict'] = data_temp[predicted_feature].shift(-1)
                print(data_temp.head())
                print(data_temp.dtypes)

                data[datasets_folder] = data_temp

    # data['classification_to_predict'] = list(map(classify, data['close'], data['future']))

    return data


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

    a = np.random.shuffle(t_minus_window_data)

    return a


def data_labeling(data, features, timestamp_size):
    x_data = createTimeWindows(data[features], timestamp_size=timestamp_size)
    y_data = data[['value_to_predict']].to_numpy()
    return x_data, y_data[timestamp_size - 1:]


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
    # lista = []
    # for i in range(4):
    #     lista.append(np.array([np.array([i, i+1, i+2]),
    #                           np.array([i, i+4, i+5])]
    #                           ))
    # lt = np.array(lista)
    # b=1

    features = ['volumefromNorm', 'volumetoNorm']
    predicted_feature = 'midPriceNorm'
    timestamp_size = 72
    df = read_data(folder_loc="../PreProcessing/cleaned_data", features=features, predicted_feature=predicted_feature)
    # plot_time_series(data=df['complete'], time_range=500, variable=['midPriceNorm'])
    x_train, y_train = data_labeling(data=df["training_set"], features=features, timestamp_size=timestamp_size)
    x_train = x_train[:10, :, :]
    y_train = y_train[:10, :]
    validation_data = data_labeling(data=df["validation_set"], features=features, timestamp_size=timestamp_size)
    a = validation_data[0][:validation_data[0].shape[0] - 10, :, :]
    b = validation_data[1][:validation_data[1].shape[0] - 10, :]
    validation_data = (a, b)
    model = create_lstm_model(data=x_train)
    training_history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=validation_data)
    plot_loss(training_history)

    x_test = validation_data[0][:-10]
    y_test = validation_data[1][:-10]
    y_predicted = model.predict(x_test)
    plot_predict(y_test, y_predicted)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predict(real, predicted):
    plt.figure(figsize=(20, 10))
    plt.plot(real, color='blue', label='Bitcoin')
    plt.plot(predicted, color='red', label='Predicted Bitcoin')
    plt.xlabel('Time: Hourly ')
    plt.ylabel('Bitcoin Market price')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
