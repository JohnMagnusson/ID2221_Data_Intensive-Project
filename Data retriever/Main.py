from Client import Client
from DataManager import saveData
from DataObjects import HistoricalDataRequest, HistoricalSocialDataRequest


def main():
    dataClient = Client()

    # historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
    historicalSocialDataRequest = HistoricalSocialDataRequest(timePrefix="hour", cryptoCurrency="BTC")
    endTime = "2020-10-15 16:00:00"
    startTime = "2019-03-12 05:00:00"

    # dataList = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
    dataList = dataClient.getHistoricalSocialDataBetween(historicalSocialDataRequest, startTime, endTime)

    saveData(dataList, fileName="bitcoin_social_max_raw", format="json")


if __name__ == "__main__":
    main()
