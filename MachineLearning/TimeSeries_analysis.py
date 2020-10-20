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
    :param is_predict_multiple: A flag, if we want to predict multiple features or nor
    :param timestamp_size: How many previous data to consider
    :param data: training data
    :return: (nr_data_samples-window_length, window_length, nr_features)
    """

    data_x = []
    data_y = []
    for i in range(timestamp_size, (len(data) - 1)):
        data_x.append(data[i - timestamp_size:i, :data.shape[1] - 1])
        if is_predict_multiple:
            data_y.append(data[i, :data.shape[1] - 1])
        else:
            data_y.append(data[i, -1])
    return np.array(data_x), np.array(data_y)


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
    """
    Creates a simple LSTM model
    :param data: The data to get the input shape from
    :param num_features: Features that the data have or will predict if is_predict_multiple flag is true
    :param is_predict_multiple: Flag, if the model should predict multiple features
    :return:
    """

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
    """
    Plots loss from training
    :param history: Loss dict
    :return:
    """

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MSE]')
    plt.legend()
    plt.grid(True)
    plt.show()


def forecast(model, from_data_sequence, steps_into_future):
    """
    Forecast the value of the cryptocurrency steps_into_future
    :param model: The trained model used to make the forecast
    :param from_data_sequence: The data sequence data the network will make the forecast from
    :param steps_into_future: How many steps into the future the forecast should have
    :return: An numpy array with the prediction into the future
    """

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
    """
    Plots the prediction and real value in a graph
    :param real: The ground truth how the value
    :param predicted: The predicted value
    :param plot_all: If we should plot all predicted values if there are more than one
    :return:
    """

    plt.figure(figsize=(20, 10))
    if plot_all:
        plt.plot(real, color='blue', label='Bitcoin')
        plt.plot(predicted, color='red', label='Predicted Bitcoin')
    else:
        if predicted.shape[1] > 1:
            plt.plot(predicted[:, 0], color='red', label='Predicted Bitcoin')
        else:
            plt.plot(predicted, color='red', label='Predicted Bitcoin')
        # if real.shape[1] > 1:
        #     plt.plot(real[:, 0], color='blue', label='Bitcoin')
        # else:
        plt.plot(real, color='blue', label='Bitcoin')
    plt.xlabel('Time: Hourly ')
    plt.ylabel('Bitcoin Market price on 0-1 scale')
    plt.legend()
    plt.ylim([0, 1])
    plt.show()


def main():
    epochs = 1  # Number of epochs to do during training
    batch_size = 128  # batch size for training
    timeWindow = 24  # How many time units will be used to make an prediction
    testing_samples = 100  # The number of samples to predict on
    # features = ['volumetoNorm', 'pageViewsNorm', 'fbTalkingNorm', 'redditPostsNorm', 'redditCommentsNorm']
    # features = ['volumetoNorm']
    features = []  # The features the model will use to make an prediction
    predicted_feature = 'midPriceNorm'  # The feature that we want to predict
    predict_multiple_features = True  # If we want to predict multiple features, if so input features = output features

    total_features = copy.deepcopy(features)
    total_features.append(predicted_feature)
    number_features = len(total_features)

    df = read_data("../PreProcessing/cleaned_data", features=features, predicted_feature=predicted_feature,
                   type_of_data="complete")
    # df = read_data("../PreProcessing/cleaned_data_socialMedia", features=features, predicted_feature=predicted_feature,
    #                type_of_data="socialMedia")

    # plot_parameter(df['training_set'], predicted_feature)

    # Get training data
    train_x, train_y = data_labeling(data=np.array(df["training_set"]), features=total_features,
                                     timestamp_size=timeWindow,
                                     testing_size=None,
                                     predict_multiple_features=predict_multiple_features)

    # Get validation and test data
    validation, (test_x, test_y) = data_labeling(data=np.array(df["validation_set"]),
                                                 features=total_features, timestamp_size=timeWindow,
                                                 testing_size=testing_samples,
                                                 predict_multiple_features=predict_multiple_features)

    # Create the model
    regressor = create_lstm_model(data=train_x, num_features=number_features,
                                  is_predict_multiple=predict_multiple_features)

    # Train the model
    regressor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=validation, shuffle=True)

    # Predicts n step into future. Only works when we have the same number of feature inputs as output
    if predict_multiple_features:
        predicted_sequence = forecast(regressor, test_x[timeWindow:], steps_into_future=100)
        plot_predict(test_y, predicted_sequence, plot_all=False)

    # Predicts one step on each row in batch
    predicted_testing_data = regressor.predict(test_x)
    plot_predict(test_y, predicted_testing_data, plot_all=False)


if __name__ == "__main__":
    main()
