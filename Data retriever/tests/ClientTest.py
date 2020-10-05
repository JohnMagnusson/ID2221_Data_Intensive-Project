import unittest

from Client import *

client = Client()

# Todo fix the problem of having inconsistent conversion between time! WTF!?=#"ODJNAKJFNKLJFJLNK

class MyTestCase(unittest.TestCase):

    def test_something(self):
        self.assertEqual(True, True)

    def test_getUpToTimeStampsDay(self):
        startTime = "2018-10-29 13:55:26"
        endTime = "2018-10-29 18:55:26"
        timePrefix = "day"
        windows = client.getUpToTimeStamps(startTime, endTime, timePrefix)
        print windows
        self.assertEqual(len(windows), 1)

    def test_getUpToTimeStampsHour(self):
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

    def test_getUpToTimeStampsMinute(self):
        pass


if __name__ == '__main__':
    unittest.main()
