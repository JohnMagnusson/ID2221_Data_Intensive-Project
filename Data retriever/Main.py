from Client import Client
from DataManager import saveData
from DataObjects import HistoricalDataRequest, HistoricalSocialDataRequest


def main():
    dataClient = Client()

    historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
    historicalSocialDataRequest = HistoricalSocialDataRequest(timePrefix="hour", cryptoCurrency="BTC")
    # endTime = "2020-10-29 15:55:26"
    # startTime = "2014-01-01 12:00:00"
    endTime = "2020-10-11 12:00:00"
    startTime = "2020-10-10 12:00:00"
    # dataList = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
    dataList = dataClient.getHistoricalSocialDataBetween(historicalSocialDataRequest, startTime, endTime)

    saveData(dataList, fileName="bitcoin_social_json", format="json")


if __name__ == "__main__":
    main()
