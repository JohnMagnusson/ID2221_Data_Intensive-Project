from Client import Client
from DataManager import saveData
from DataObjects import HistoricalDataRequest


def main():
    dataClient = Client()

    historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
    startTime = "2018-10-29 13:55:26"
    endTime = "2018-10-29 15:55:26"
    dataList = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
    print("Retrieved this data from the API: ", dataList)
    saveData(dataList, fileName="test1")


if __name__ == "__main__":
    main()
