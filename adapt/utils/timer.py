from datetime import datetime
from datetime import timedelta

class Timeout(Exception):
  '''An exception that raised by Timer when the set time budget is expired.'''
  pass

class Timer:
  '''A '''

  def __init__(self, hours=0, minutes=0, seconds=0):
    '''Create a timer with time budget.

    Args:
      hours: A non-negative integer which indicates the time budget in hours.
        0 for the default value.
      minutes: A non-negative integer which indicates the time budget in minutes.
        0 for the defalut value.
      seconds: A non-negative integer which indicates the time budget in seconds.
        0 for the defalut value. If all 3 arguments are set to be 0, the time budget
        will automatically set to be 10 seconds.

    Raises:
      ValueError: When one of the arguments is negative.

    Example:

    >>> from adapt.utils.timer import Timer
    >>> timer = Timer(minutes=10, seconds=30) # This will create a timer with the time budget of 1 minute and 30seconds.
    '''

    # Check parameters.

    # Set time budget using timedelta.
    self.time_budget = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    # Set start time as the creation time.
    self.start_time = datetime.now()

  def check_timeout(self):
    '''Check whether time budget is expired or not.

    Raises:
      Timeout: When time budget is expired.

    Example:

    >>> from adapt.utils.timer import Timeout
    >>> from adapt.utils.timer import Timer
    >>> from time import sleep
    >>> timer = Timer(seconds=5)
    >>> try:
    ...   t = 0
    ...   while True:
    ...     sleep(1)
    ...     t += 1
    ...     print('{} seconds passed.'.format(t))
    ...     timer.check_timeout()
    ... except Timeout:
    ...   print('Timeout!')
    ...
    1 seconds passed.
    2 seconds passed.
    3 seconds passed.
    4 seconds passed.
    5 seconds passed.
    Timeout!
    '''

    # Check if time budget is expired.
    if datetime.now() - self.start_time > self.time_budget:
      raise Timeout()

  @property
  def elapsed(self):
    '''A `datetime.timedelta` object that indicates the time elapsed after the creation of Timer.

    Example:

    >>> from adapt.utils.timer import Timer
    >>> timer = Timer(minutes=1)
    >>> # After a few seconds
    >>> int(timer.elaped.total_seconds())
    13
    '''

    return datetime.now() - self.start_time
