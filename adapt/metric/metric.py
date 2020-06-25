from abc import ABC
from abc import abstractmethod

class Metric(ABC):
  '''Abstract metric class (used as an implementation base).'''

  def __init__(self):
    '''Create a metric.
    
    *** This method could be updated, but not mandatory. ***
    '''

  def __call__(self, **kwargs):
    '''Python magic call method.
    
    This will make object callable. Just passing the arguments to covered method.
    '''

    return self.covered(**kwargs)

  @abstractmethod
  def covered(self, **kwargs):
    '''Gets output of network and returns a list of corresponding coverage vectors.
    
    *** This method should be implemented. ***

    Args:
      kwargs: A dictionary of keyword arguments. The followings are privileged
        arguments.
      internals: A list of the values of internal neurons in each layer.
      logits: Output logits.

    Returns:
      A list of coverage vectors that identifies which neurons are activated.
    '''
