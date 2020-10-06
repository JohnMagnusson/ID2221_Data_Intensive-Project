from __future__ import division  # Override python 2.x division that produces int to 3.x which uses floats

import json
import math
from tqdm import tqdm

import requests

from Utils import *


class Client:
    def __init__(self):
        self.apiKey = "2caa284366c7ebc9340e5ce55bbf16de4fae6aa49b2b0ec93eac7c2ff385af95"
        self.endpoint = "https://min-api.cryptocompare.com/data/v2/"
        self.dataWindowSize = 2000  # 2000 is the max the API allows

    def getHistoricalDataBetween(self, requestParameters, startTime, endTime):
        # Divide the number of toTimeStamp by dividing by the time/(timePrefix * 2000)
        windowsSettings = self.getUpToTimeStamps(startTime, endTime, requestParameters.timePrefix)
        dataBataches = []

        print("Getting data with the set window")
        for windowSetting in tqdm(windowsSettings):
            dataBataches.append(self.getHistoricalData(requestParameters.timePrefix, requestParameters.cryptoCurrency,
                                                       requestParameters.fiatCurrency, windowSetting[0], windowSetting[1]))
        print("Done fetching data!")
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
        if timePrefix == "hour":
            return 3600
        elif timePrefix == "day":
            return 86400
        elif timePrefix == "minute":
            return 60
        else:
            raise Exception("Invalid timePrefix")

    def getHistoricalData(self, timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", toTimeStamp=-1, nrRecordsToGet=10):
        finalUrl = self.endpoint + "histo" + timePrefix + "?fsym=" + cryptoCurrency + "&tsym=" + fiatCurrency + "&limit=" + str(nrRecordsToGet-1)   # We do nrRecordsToGet-1 because they start at 0
        if toTimeStamp != -1:
            finalUrl = finalUrl + "&toTs=" + str(toTimeStamp)

        # auth=HTTPDigestAuth("authorization", self.apiKey)
        # response = requests.get(finalUrl, auth=("authorization", self.apiKey))
        response = requests.get(finalUrl)
        jData = json.loads(response.content)
        return jData
