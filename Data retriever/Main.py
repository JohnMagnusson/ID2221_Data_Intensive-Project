from Client import Client
from DataManager import saveData
from DataObjects import HistoricalDataRequest


def main():
    dataClient = Client()

    historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
    endTime = "2020-10-29 15:55:26"
    startTime = "2014-01-01 12:00:00"
    dataList = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
    saveData(dataList, fileName="bitcoin_json", format="json")


if __name__ == "__main__":
    main()
