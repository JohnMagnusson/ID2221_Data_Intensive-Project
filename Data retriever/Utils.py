import time
from datetime import datetime


def stringToTimeStamp(timeStr):
    """
    Important to know that it returns the time in UNIX UTC format!
    :param timeStr: String in format of year-month-day hour:minute:second
    :return: Timestamp in unix format in UTC
    """
    timeStr = timeStr + "-UTC"
    date_object = datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S-%Z')
    return time.mktime(date_object.timetuple())# + 7200  # Add offset because we live in Sweden an convert it to UTC.
    # Please someone save me from this date/time hell


def timeStampToDateTime(timeStamp):
    """
    Converts unix to unix time
    :param timeStamp: Timestamp in unix format
    :return: The input in dateTime format in UTC
    """
    return datetime.utcfromtimestamp(timeStamp)
