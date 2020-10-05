
class HistoricalDataRequest:
    def __init__(self, timePrefix = "histohour", cryptoCurrency = "BTC",  fiatCurrency = "USD"):
        self.timePrefix = timePrefix
        self.cryptoCurrency = cryptoCurrency
        self.fiatCurrency = fiatCurrency