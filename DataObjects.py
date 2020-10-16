class HistoricalDataRequest:
    def __init__(self, timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD"):
        self.timePrefix = timePrefix
        self.cryptoCurrency = cryptoCurrency
        self.fiatCurrency = fiatCurrency


class HistoricalSocialDataRequest:
    def __init__(self, timePrefix="hour", cryptoCurrency="BTC"):
        self.timePrefix = timePrefix
        self.cryptoCurrency = cryptoCurrency
