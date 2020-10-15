import unittest

from Client import *

client = Client()

class MyTestCase(unittest.TestCase):
    # Following tests controls that the windowing of timeStamps to use to the API adds up correctly
    def test_getUpToTimeStampsDaySmallTimeDifference(self):
        startTime = "2018-10-27 18:55:26"
        endTime = "2018-10-29 18:55:26"
        timePrefix = "day"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        correctList = []
        correctList.append((1540839326.0, 2))
        self.assertEqual(correctList, windows)

    def test_getUpToTimeStampsDayBigTimeDifference(self):
        startTime = "2013-10-27 18:55:26"
        endTime = "2020-10-29 18:55:26"
        timePrefix = "day"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        self.assertEqual(2, len(windows))
        self.assertEqual((1431197726.0, 559), windows[-1])

    def test_getUpToTimeStampsHourSmallTimeDifference(self):
        startTime = "2018-10-28 13:55:26"
        endTime = "2018-10-28 18:55:26"
        timePrefix = "hour"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        correctList = []
        correctList.append((1540752926.0, 5))
        self.assertEqual(correctList, windows)

    def test_getUpToTimeStampsHourBigTimeDifference(self):
        startTime = "2016-10-29 13:55:26"
        endTime = "2020-10-05 16:55:26"
        timePrefix = "hour"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        self.assertEqual(18, len(windows))
        self.assertEqual((1479516926.0, 491), windows[-1])

    def test_getUpToTimeStampsMinuteSmallTimeDifference(self):
        startTime = "2018-10-28 18:50:00"
        endTime = "2018-10-28 18:55:30"
        timePrefix = "minute"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        correctList = []
        correctList.append((1540752930.0, 5))
        self.assertEqual(correctList, windows)

    def test_getUpToTimeStampsMinuteBigTimeDifference(self):
        startTime = "2012-10-29 13:55:45"
        endTime = "2020-03-05 16:55:26"
        timePrefix = "minute"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)

        self.assertEqual(1933, len(windows))
        self.assertEqual((1351587326.0, 1139), windows[-1])


    # def test_getHistoricalDataWithTimeStamp(self):
    #     nrRecordsToGet = 3
    #     timeStamp = 1543496126.0
    #     data = client.getHistoricalData(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", toTimeStamp=timeStamp, nrRecordsToGet=nrRecordsToGet)
        # self.assertFalse(data[u'HasWarning'])
        # self.assertEqual(data[u'Type'], 100)
        # self.assertEqual(data[u'Response'], u'Success')
        # self.assertEqual(data[u'Response'], u'Success')
        # self.assertEqual(len(data[u'Data'][u'Data']), nrRecordsToGet)
        # isInTimeScope = data[u'Data'][u'TimeTo'] <= timeStamp    # We check that the latest retreived timeStamp is inside the time scope have given
        # self.assertTrue(isInTimeScope)

    # def test_getHistoricalDataWithoutTimeStamp(self):
    #     nrRecordsToGet = 3
    #     data = client.getHistoricalData(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", nrRecordsToGet=nrRecordsToGet)
    #     self.assertFalse(data[u'HasWarning'])
    #     self.assertEqual(data[u'Type'], 100)
    #     self.assertEqual(data[u'Response'], u'Success')
    #     self.assertEqual(data[u'Response'], u'Success')
    #     self.assertEqual(len(data[u'Data'][u'Data']), nrRecordsToGet)

    def test_assertOrderInGetHistoricalData(self):
        nrRecordsToGet = 2
        data = client.getHistoricalData(timePrefix="hour", cryptoCurrency="BTC", fiatCurrency="USD", nrRecordsToGet=nrRecordsToGet)
        keys_order = ['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 'conversionType', 'conversionSymbol']

        for i in range(len(data)):
            dict_keys = list(data[i].keys())
            for key in range(len(data[i].keys())):
                self.assertEqual(dict_keys[key], keys_order[key])

    def test_coinNameToCoinIdBTC(self):
        coinId = client.coinNameToCoinId("BTC")
        self.assertEqual("1182", coinId)


    def test_getHistoricalSocialDataWithoutTimeStamp(self):
        nrRecordsToGet = 2
        data = client.getHistoricalSocialData(timePrefix="hour", cryptoCurrency="BTC", nrRecordsToGet=nrRecordsToGet)
        self.assertEqual(nrRecordsToGet, len(data))

    def test_getHistoricalSocialDataWithTimeStamp(self):
        nrRecordsToGet = 3
        timeStamp = 1543496126.0
        data = client.getHistoricalSocialData(timePrefix="day", cryptoCurrency="BTC", toTimeStamp=timeStamp, nrRecordsToGet=nrRecordsToGet)
        self.assertEqual(nrRecordsToGet, len(data))

    def test_assertOrderInGetHistoricalSocialData(self):
        nrRecordsToGet = 2
        data = client.getHistoricalSocialData(timePrefix="hour", cryptoCurrency="BTC", nrRecordsToGet=nrRecordsToGet, toTimeStamp=1637880400)
        keys_order = ['time', 'comments', 'posts', 'followers', 'points', 'overview_page_views', 'analysis_page_views', 'markets_page_views',
                      'charts_page_views', 'trades_page_views', 'forum_page_views', 'influence_page_views', 'total_page_views',
                      'fb_likes', 'fb_talking_about', 'twitter_followers', 'twitter_following', 'twitter_lists', 'twitter_favourites',
                      'twitter_statuses', 'reddit_subscribers', 'reddit_active_users', 'reddit_posts_per_hour', 'reddit_posts_per_day',
                      'reddit_comments_per_hour', 'reddit_comments_per_day', 'code_repo_stars', 'code_repo_forks', 'code_repo_subscribers',
                      'code_repo_open_pull_issues', 'code_repo_closed_pull_issues', 'code_repo_open_issues', 'code_repo_closed_issues',
                      'code_repo_contributors']

        for i in range(len(data)):
            dict_keys = list(data[i].keys())
            for key in range(len(data[i].keys())):
                self.assertEqual(dict_keys[key], keys_order[key])

if __name__ == '__main__':
    unittest.main()
