import unittest

from Utils import *


class MyTestCase(unittest.TestCase):
    def test_stringToTimeStamp(self):
        exampleDate = "2018-11-29 13:55:26"
        timeStamp = stringToTimeStamp(exampleDate)
        self.assertEqual(1543499726.0, timeStamp)

    def test_stringToTimeStamp_2(self):
        exampleDate = "2020-10-05 16:55:26"
        timeStamp = stringToTimeStamp(exampleDate)
        self.assertEqual(1601916926.0, timeStamp)

    def test_stringToTimeStamp_3(self):
        exampleDate = "2016-10-29 13:55:26"
        timeStamp = stringToTimeStamp(exampleDate)
        self.assertEqual(1477749326.0, timeStamp)

    def test_stringToTimeStamp_4(self):
        exampleDate = "2007-01-01 00:00:00"
        timeStamp = stringToTimeStamp(exampleDate)
        self.assertEqual(1167609600.0, timeStamp)

    def test_timeStampToDateTime(self):
        timestamp = 1543496126.0
        convertedTime = timeStampToDateTime(timestamp)
        correctConvert = datetime.strptime("2018-11-29 12:55:26", '%Y-%m-%d %H:%M:%S')
        self.assertEqual(correctConvert, convertedTime)


if __name__ == '__main__':
    unittest.main()
