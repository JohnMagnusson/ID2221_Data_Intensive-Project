# packages and libraries
import copy
import os
import random

import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import Sequential

tensorflow.random.set_seed(12)
np.random.seed(12)
random.seed(12)

## TODO:
# - In the createTimeWindows removes the last feature. Previous it was time but now it can be anything so we need to fix this
# - There is a nan in the test or validation data, do we need to fix it? If how, remove or add values? (I saw this in the social data)
# - When predicting multiple features and we only want to plot one of them it is kinda wonky. Do we care?
# - why is y[t] != to x[t-1]?
# - try to predict mid price [t] without mid price [t-1, -2, ...]


# Allows to run on GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def read_data(folder_loc, features, predicted_feature, type_of_data):
    """
    :param type_of_data:
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
                if type_of_data == "complete":
                    data_temp = pd.DataFrame(pd.read_csv(data_loc, sep=',', index_col=False,
                                                         names=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto',
                                                                'close', 'converstionType', 'conversionSymbol',
                                                                'midPrice', 'midPriceNorm', 'volumefromNorm',
                                                                'volumetoNorm', 'empty']))
                if type_of_data == "socialMedia":
                    data_temp = pd.DataFrame(pd.read_csv(data_loc, sep=',', index_col=False,
                                                         names=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto',
                                                                'close', 'converstionType', 'conversionSymbol',
                                                                'midPrice', 'midPriceNorm', 'volumefromNorm',
                                                                'volumetoNorm', 'total_page_views', 'fb_talking_about',
                                                                'reddit_posts_per_hour', 'reddit_comments_per_hour',
                                                                'pageViewsNorm', 'fbTalkingNorm', 'redditPostsNorm',
                                                                'redditCommentsNorm', 'empty']))

                data_temp["time"] = data_temp["time"].str[1:].astype(float)

                data_temp.set_index("time", inplace=True)

                if features is not None:
                    data_temp = data_temp[features]

                data_temp['value_to_predict'] = data_temp[predicted_feature].shift(-1)
                print(data_temp.head())
                print(data_temp.dtypes)

                data[datasets_folder] = data_temp

    return data


def createTimeWindows(data, timestamp_size, is_predict_multiple):
    """
    We should consider the the previous data when predicting. In this case with a window_length size
    :param timestamp_size: How many previous data to consider
    :param data: training data
    :return: (nr_data_samples-window_length, window_length, nr_features)
    """

    data_x = []
    data_y = []
    for i in range(timestamp_size, (len(data) - 1)):
        data_x.append(data[i - timestamp_size:i, :data.shape[
                                                      1] - 1])  # TODO Here we specify -1 to remove time but actually we loose the last feature
        if is_predict_multiple:
            data_y.append(data[i, :data.shape[1] - 1])
        else:
            data_y.append(data[i])
    # Takes two list and shuffles them together in order
    zipped = list(zip(data_x, data_y))
    random.shuffle(zipped)
    data_x_shuffled, data_y_shuffled = zip(*zipped)
    return np.array(data_x_shuffled), np.array(data_y_shuffled)


def data_labeling(data, features, timestamp_size, testing_size=None, predict_multiple_features=False):
    if testing_size is not None:
        x, y = createTimeWindows(data, timestamp_size=timestamp_size, is_predict_multiple=predict_multiple_features)
        validation_x = x[:-testing_size]
        validation_y = y[:-testing_size]
        test_x = x[-testing_size:]
        test_y = y[-testing_size:]
        return (validation_x, validation_y), (test_x, test_y)
    else:
        train_x, train_y = createTimeWindows(data, timestamp_size=timestamp_size,
                                             is_predict_multiple=predict_multiple_features)
        return train_x, train_y


def create_lstm_model(data, num_features, is_predict_multiple=False):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1], num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    if is_predict_multiple:
        model.add(Dense(units=num_features))
    else:
        model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=MeanSquaredError())
    return model


def plot_parameter(data, parameter):
    plt.plot(data[parameter], label=parameter)
    plt.grid(True)
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show()


def forecast(model, from_data_sequence, steps_into_future):
    predicted_sequence = []
    data_sequence = from_data_sequence[-1]
    window_length = from_data_sequence.shape[1]
    for i in range(steps_into_future):
        tmp = data_sequence[-window_length:].reshape((1, window_length, data_sequence.shape[1]))
        predicted_value = model.predict(tmp)
        if predicted_value.shape[1] != data_sequence.shape[1]:
            raise Exception(
                "To make a forecast the number of features outputted  from the model most be the same number as input")
        data_sequence = np.concatenate((data_sequence, predicted_value))
        predicted_sequence.append(predicted_value[0])
    return np.array(predicted_sequence)


def plot_predict(real, predicted, plot_all=False):
    plt.figure(figsize=(20, 10))
    if plot_all:
        plt.plot(real, color='blue', label='Bitcoin')
        plt.plot(predicted, color='red', label='Predicted Bitcoin')
    else:
        if predicted.shape[1] > 1:
            plt.plot(predicted[:, 0], color='red', label='Predicted Bitcoin')
        else:
            plt.plot(predicted, color='red', label='Predicted Bitcoin')
        if real.shape[1] > 1:
            plt.plot(real[:, 0], color='blue', label='Bitcoin')
        else:
            plt.plot(real, color='blue', label='Bitcoin')
    plt.xlabel('Time: Hourly ')
    plt.ylabel('Bitcoin Market price on 0-1 scale')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()


def main():
    epochs = 1
    batch_size = 128
    timeWindow = 7
    testing_samples = 100
    features = ['volumetoNorm', 'pageViewsNorm', 'fbTalkingNorm', 'redditPostsNorm', 'redditCommentsNorm']
    predicted_feature = 'midPriceNorm'
    predict_multiple_features = False

    total_features = copy.deepcopy(features)
    total_features.append(predicted_feature)
    number_features = len(total_features)

    # df = read_data("../PreProcessing/cleaned_data", features=features, predicted_feature=predicted_feature,
    #                type_of_data="complete")
    df = read_data("../PreProcessing/cleaned_data_socialMedia", features=features, predicted_feature=predicted_feature,
                   type_of_data="socialMedia")

    # plot_parameter(df['training_set'], predicted_feature)

    train_x, train_y = data_labeling(data=np.array(df["training_set"]), features=total_features,
                                     timestamp_size=timeWindow,
                                     testing_size=None,
                                     predict_multiple_features=predict_multiple_features)

    validation, (test_x, test_y) = data_labeling(data=np.array(df["validation_set"]),
                                                 features=total_features, timestamp_size=timeWindow,
                                                 testing_size=testing_samples,
                                                 predict_multiple_features=predict_multiple_features)

    regressor = create_lstm_model(data=train_x, num_features=number_features,
                                  is_predict_multiple=predict_multiple_features)
    regressor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=validation, shuffle=True)

    # Predicts n step into future. Only works when we have the same number of feature inputs as output
    if predict_multiple_features:
        predicted_sequence = forecast(regressor, test_x[timeWindow:], steps_into_future=testing_samples)
        plot_predict(test_y, predicted_sequence, plot_all=False)

    # Predicts one step on each row in batch
    predicted_testing_data = regressor.predict(test_x)
    plot_predict(test_y, predicted_testing_data, plot_all=False)


if __name__ == "__main__":
    main()
