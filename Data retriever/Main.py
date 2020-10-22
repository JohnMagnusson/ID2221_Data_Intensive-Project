from Client import Client
from DataManager import saveData
from DataObjects import HistoricalDataRequest, HistoricalSocialDataRequest


def main():
    dataClient = Client()

    get_social_data = True
    get_trading_data = True

    # Between which time periods one wants to fetch data
    endTime = "2020-10-15 16:00:00"
    startTime = "2019-03-12 05:00:00"

    if get_trading_data:
        historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
        dataList = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
        saveData(dataList, fileName="bitcoin_trading_data", format="json")

    if get_social_data:
        historicalSocialDataRequest = HistoricalSocialDataRequest(timePrefix="hour", cryptoCurrency="BTC")
        dataList = dataClient.getHistoricalSocialDataBetween(historicalSocialDataRequest, startTime, endTime)
        saveData(dataList, fileName="bitcoin_social_data", format="json")


if __name__ == "__main__":
    main()
