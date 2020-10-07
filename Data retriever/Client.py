from __future__ import division  # Override python 2.x division that produces int to 3.x which uses floats

import json
import math
from tqdm import tqdm

import requests

from Utils import *


class Client:
    def __init__(self):
        self.apiKey = "2caa284366c7ebc9340e5ce55bbf16de4fae6aa49b2b0ec93eac7c2ff385af95"
        self.useApiKey = False
        self.endpoint = "https://min-api.cryptocompare.com/data/v2/"
        self.dataWindowSize = 2000  # 2000 is the max the API allows

    def getHistoricalDataBetween(self, requestParameters, startTime, endTime):
        """
        Gets historical data between startTime and endTime given the requestParameters
        :param requestParameters: Request parameters
        :param startTime: From what time forward data should be retrieved
        :param endTime: The maximum time data should be retrieved
        :return: A list of Json objects from the API
        """

        windowsSettings = self.getUpToTimeStamps(startTime, endTime, requestParameters.timePrefix)
        dataBatches = []

        print("Getting data with the set window")
        for windowSetting in tqdm(windowsSettings):
            dataBatches.append(self.getHistoricalData(requestParameters.timePrefix, requestParameters.cryptoCurrency,
                                                       requestParameters.fiatCurrency, windowSetting[0], windowSetting[1]))
        print("Done fetching data!")
        return dataBatches

    def getUpToTimeStamps(self, startTime, endTime, timePrefix):
        """
        Returns a list of timeStamps to get all the data betwen starTime end endTime with a windowing defined by timePrefix
        :param startTime: Str
        :param endTime: Str
        :param timePrefix: Str
        :return: list(timeStamps, windowSize) -> list(String, Int)
        """
        tsStartTime = stringToTimeStamp(startTime)
        tsEndTime = stringToTimeStamp(endTime)

        # To verify that the input parameters are correct
        if tsEndTime < tsStartTime:
            raise Exception("endTime must be later than the startTime!")

        intTimePrefix = self.timePrefixToInteger(timePrefix)
        nrWindows = int(math.ceil((tsEndTime - tsStartTime) / (intTimePrefix * self.dataWindowSize)))

        upToBatches = []
        for i in range(nrWindows):
            ts = tsEndTime - (i * intTimePrefix * self.dataWindowSize)
            if i == nrWindows - 1 and len(
                    upToBatches) == 0:  # last one is special case where we want to limit the windows size
                # We round down as that batch wont ce computed by the API yet
                windowSize = int(math.floor((tsEndTime - tsStartTime) / intTimePrefix))
            elif i == nrWindows - 1:
                # We round down as that batch wont ce computed by the API yet
                windowSize = int(math.floor((upToBatches[-1][0] - tsStartTime) / intTimePrefix)) - self.dataWindowSize
            else:
                windowSize = self.dataWindowSize

            upToBatches.append((ts, windowSize))
        return upToBatches

    def timePrefixToInteger(self, timePrefix):
        """
        Convert string time prefix to time in seconds
        :param timePrefix: The prefix of the time (day, hour, minute)
        :return: int
        """
        if timePrefix == "hour":
            return 3600
        elif timePrefix == "day":
            return 86400
        elif timePrefix == "minute":
            return 60
        else:
            raise Exception("Invalid timePrefix")

    def getHistoricalData(self, timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", toTimeStamp=-1, nrRecordsToGet=10):
        """
        Gets the historical data with the input paramters
        :param timePrefix: The prefix of the time (day, hour, minute)
        :param cryptoCurrency: Which cryptocurrency to get data on (BTC, ETH etc.)
        :param fiatCurrency: Which currency to compare the price with (USD, GBP)
        :param toTimeStamp: Up to witch time stamp data should be retrieved
        :param nrRecordsToGet: Number of records to get up to the toTimeStamp
        :return: A json containing the data from the API
        """

        finalUrl = self.endpoint + "histo" + timePrefix + "?fsym=" + cryptoCurrency + "&tsym=" + fiatCurrency + "&limit=" + str(nrRecordsToGet-1)   # We do nrRecordsToGet-1 because they start at 0
        if toTimeStamp != -1:
            finalUrl = finalUrl + "&toTs=" + str(toTimeStamp)

        # We do not need to use the key always and we have a limited number of calls to use for free
        if self.useApiKey:
            response = requests.get(finalUrl, auth=("authorization", self.apiKey))
        else:
            response = requests.get(finalUrl)

        isProcessedCorrectly, errorMessage = self.validateGoodHttpResponse(response.status_code)
        if not isProcessedCorrectly:
            raise Exception(errorMessage)
        return json.loads(response.content)    # Converts data to JSON

    def validateGoodHttpResponse(self, status_code):
        """
        Calidates that the http stats_code is okay. If not return false and an error message
        :param status_code: HTTP status code
        :return: (bool, String) if the request was processed correctly, if not an error code with it
        """
        if status_code == 200:
            return True, "Http request was successful"
        elif status_code == 404:
            return False, "Http URL was wrongly formatted please check over url"
        elif status_code == 408:
            return False, "The endpoint timed out, double check format at try again"
        elif status_code == 500:
            return False, "Endpoint could not handle the request, look over input variabels"
        else:
            return False, "Non supported HTTP response code was received: " + str(status_code) + " closing program."

