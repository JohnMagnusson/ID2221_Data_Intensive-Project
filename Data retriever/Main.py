from Client import Client
from DataObjects import HistoricalDataRequest


def main():
    dataClient = Client()

    historicalDataRequest = HistoricalDataRequest(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD")
    startTime = "2018-10-29 13:55:26"
    endTime = "2018-10-29 15:55:26"
    data = dataClient.getHistoricalDataBetween(historicalDataRequest, startTime, endTime)
    print("data: ", data)


if __name__ == "__main__":
    main()
