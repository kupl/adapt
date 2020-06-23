import numpy as np

from adapt.metric.metric import Metric

class NeuronCoverage(Metric):
  '''Neuron coverage (NC).
  
  Neuron coverage is a coverage metric that identifies the neurons with values
  that are higher than a certain threshold (theta). Please, see the following
  paper for more details:
  
  DeepXplore: Automated Whitebox Testing of Deep Learning Systems
  https://arxiv.org/abs/1705.06640
  '''

  def __init__(self, theta=0.5):
    '''Create a neuron coverage metric with a certain threshold.
    
    Args:
      theta: An floating point value in [0, 1].

    Raises:
      ValueError: When theta is not in [0, 1].

    Example:

    >>> from adapt.metric import NC
    >>> metric = NC()
    '''

    super(NeuronCoverage, self).__init__()

    # Check the range of theta.
    if theta < 0 or theta > 1:
      raise ValueError('The argument theta is not in [0, 1].')
    self.theta = theta

  def covered(self, internals, **kwargs):
    '''Returns a list of neuron coverage vectors.
    
    Args:
      internals: A list of the values of internal neurons in each layer.
      kwargs: Not used. Present for the compatibility with the super class.
    
    Returns:
      A neuron coverage vecter that identifies which neurons have higher value
      than theta.

    Example:

    >>> from adapt.metric import NC
    >>> import tensorflow as tf
    >>> metric = NC(0.5)
    >>> internals = [tf.random.normal((3,)), tf.random.normal((2,)), tf.random.normal((3,))]
    >>> for x in internals:
    ...   print(x)
    ...
    tf.Tensor([ 1.5756989  -0.2245746  -0.40161133], shape=(3,), dtype=float32)
    tf.Tensor([-1.8598881  1.0225831], shape=(2,), dtype=float32)
    tf.Tensor([-0.2890836  1.2187911 -0.7577767], shape=(3,), dtype=float32)
    >>> covered = metric(internals=internals)
    >>> for x in covered:
    ...   print(x)
    ...
    [ True False False]
    [False  True]
    [False  True False]
    '''

    # Get numpy array.
    internals = [i.numpy() for i in internals]

    # Normalize the values into [0, 1].
    internals = [(i - np.min(i)) / (np.max(i) - np.min(i) + 1e-6) for i in internals]

    # Find neurons with the values higher than theta.
    covered = [np.where(i > self.theta, True, False) for i in internals]

    return np.array(covered)

  def __repr__(self):
    '''Returns a string representation of object.
    
    Example:
    
    >>> from adapt.metric import NC
    >>> metric = NC()
    >>> metric
    NeuronCoverage(theta=0.5)
    '''

    return 'NeuronCoverage(theta={})'.format(self.theta)
