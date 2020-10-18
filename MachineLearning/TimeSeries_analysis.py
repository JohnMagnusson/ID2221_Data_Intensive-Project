# packages and libraries
from collections import deque
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
import copy
import os
import random

random.seed(12)


## TODO:
# - is plot shifted?
# - something is random (findout)
# - estimate one week ahead without updating the model daily;
# - estimation with ground truth
# - implement validation data into the training
# - why is y[t] != to x[t-1]?


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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


def createTimeWindows(data, timestamp_size):
    """
    We should consider the the previous data when predicting. In this case with a window_length size
    :param timestamp_size: How many previous data to consider
    :param data: training data
    :return: (nr_data_samples-window_length, window_length, nr_features)
    """

    data_x = []
    data_y = []
    for i in range(timestamp_size, (len(data) - 1)):
        data_x.append(data[i - timestamp_size:i, :data.shape[1] - 1])
        data_y.append(data[i, -1])

    return np.array(data_x), np.array(data_y)

    # shape = (data.shape[0] - timestamp_size + 1, timestamp_size, data.shape[1])
    # t_minus_window_data = np.empty(shape)
    # timestamp_data = deque(maxlen=timestamp_size)
    #
    # counter = 0
    # for i in data.values:  # iterate over the values
    #     timestamp_data.append([n for n in i])  # store all but the target
    #     if len(timestamp_data) == timestamp_size:
    #         # suffled_timestamp = random.sample(timestamp_data, len(timestamp_data))
    #         suffled_timestamp = timestamp_data
    #         t_minus_window_data[counter] = np.array(suffled_timestamp)
    #         counter += 1

    # return t_minus_window_data[:-2,:,:]


def data_labeling(data, features, timestamp_size, testing_size=None):
    if testing_size is not None:
        x, y = createTimeWindows(data, timestamp_size=timestamp_size)
        validation_x = x[:-testing_size]
        validation_y = y[:-testing_size]
        test_x = x[-testing_size:]
        test_y = y[-testing_size:]
        return (validation_x, validation_y), (test_x, test_y)
    else:
        train_x, train_y = createTimeWindows(data, timestamp_size=timestamp_size)
        return train_x, train_y


def create_lstm_model(data, num_features):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(data.shape[1], num_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


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
    plt.ylabel('Bitcoin Market price in 0-1 scale')
    plt.legend()
    plt.show()


def main():
    epochs = 3
    batch_size = 128
    timeWindow = 72
    testing_samples = 1000
    features = ['volumetoNorm']
    predicted_feature = 'midPriceNorm'

    total_features = copy.deepcopy(features)
    total_features.append(predicted_feature)
    number_features = len(total_features)

    df = read_data("PreProcessing/cleaned_data", features=features, predicted_feature=predicted_feature,
                   type_of_data="complete")

    train_x, train_y = data_labeling(data=np.array(df["training_set"]), features=total_features,
                                     timestamp_size=timeWindow,
                                     testing_size=None)

    (validation_x, validation_y), (test_x, test_y) = data_labeling(data=np.array(df["validation_set"]),
                                                                   features=total_features, timestamp_size=timeWindow,
                                                                   testing_size=testing_samples)

    regressor = create_lstm_model(data=train_x, num_features=number_features)
    regressor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    predicted_testing_data = regressor.predict(test_x)

    plot_predict(test_y, predicted_testing_data)


if __name__ == "__main__":
    main()
