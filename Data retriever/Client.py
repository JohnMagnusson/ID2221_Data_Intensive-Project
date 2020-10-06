import json
import math

import requests

from Utils import *


class Client:
    def __init__(self):
        self.apiKey = "2caa284366c7ebc9340e5ce55bbf16de4fae6aa49b2b0ec93eac7c2ff385af95"
        self.endpoint = url = "https://min-api.cryptocompare.com/data/v2/"
        self.dataWindowSize = 2000  # 2000 is the max the API allows

    def getHistoricalDataBetween(self, requestParameters, startTime, endTime):
        # Divide the number of toTimeStamp by dividing by the time/(timePrefix * 2000)
        listOfTimeBatches = self.getUpToTimeStamps(startTime, endTime, requestParameters.timePrefix)
        dataBataches = []
        for batch in listOfTimeBatches:
            dataBataches.append(self.getHistoricalData(requestParameters.timePrefix, requestParameters.cryptoCurrency,
                                                       requestParameters.fiatCurrency, "5", batch))
        return dataBataches

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

        # To double check that the input parameters are correct
        if tsEndTime < tsStartTime:
            raise ("endTime must be later than the startTime!")

        intTimePrefix = self.timePrefixToInteger(timePrefix)
        nrWindows = int(math.ceil((tsEndTime - tsStartTime) / (intTimePrefix * self.dataWindowSize)))
        part1 = tsEndTime - tsStartTime
        part2 = intTimePrefix * self.dataWindowSize
        part3 = part1 / part2

        upToBatches = []
        for i in range(nrWindows):
            ts = tsEndTime - (i * intTimePrefix * self.dataWindowSize)
            if i == nrWindows - 1 and len(
                    upToBatches) == 0:  # last one is special case where we want to limit the windows size
                windowSize = int(math.ceil(
                    (tsEndTime - tsStartTime) / intTimePrefix))  # We round up, rather take to much than to little
            elif i == nrWindows - 1:
                # We round up, rather take to much then to little
                windowSize = int(math.ceil((upToBatches[-1][0] - tsStartTime) / intTimePrefix)) - self.dataWindowSize
            else:
                windowSize = self.dataWindowSize

            upToBatches.append((ts, windowSize))
        return upToBatches

    def timePrefixToInteger(self, timePrefix):
        if timePrefix == "hour":
            return 3600
        elif timePrefix == "day":
            return 86400
        elif timePrefix == "minute":
            return 60
        else:
            raise Exception("Invalid timePrefix")

    # Todo fix that we can do between period. That is ToTS works
    def getHistoricalData(self, timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", nrRecordsToGet="10",
                          toTimeStamp=""):
        finalUrl = self.endpoint + "histo" + timePrefix + "?fsym=" + cryptoCurrency + "&tsym=" + fiatCurrency + "&limit=" + nrRecordsToGet

        # auth=HTTPDigestAuth("authorization", self.apiKey)
        # response = requests.get(finalUrl, auth=("authorization", self.apiKey))
        response = requests.get(finalUrl)
        jData = json.loads(response.content)
        return jData