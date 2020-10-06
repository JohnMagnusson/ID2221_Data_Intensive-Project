# import time
import calendar
from datetime import datetime


def stringToTimeStamp(timeStr):
    """
    The conversion assume the input is in UTC and the output will also be in UTC
    :param timeStr: String in format of year-month-day hour:minute:second
    :return: Timestamp in unix format in UTC
    """
    date_object = datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
    return calendar.timegm(date_object.timetuple())


def timeStampToDateTime(timeStamp):
    """
    Converts unix to unix time
    :param timeStamp: Timestamp in unix format
    :return: The input in dateTime format in UTC
    """
    return datetime.utcfromtimestamp(timeStamp)
