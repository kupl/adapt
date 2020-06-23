import numpy as np

from adapt.metric.metric import Metric

class TopkNeuronCoverage(Metric):
  '''Tok-k Neuron Coverage (TKNC).
  
  Top-k neuron coverage is a coverage metric that identifies the neuron within
  the highest k-th values in their layers. Please, see the following paper for
  more details:

  DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems
  https://arxiv.org/abs/1803.07519
  '''

  def __init__(self, k=3):
    '''Create a top-k neuron coverage metric with a certain k.

    Args:
      k: A positive integer.

    Raises:
      ValueError: When k is not positive.

    Example:

    >>> from adapt.metric import TKNC
    >>> metric = TKNC()
    '''

    super(TopkNeuronCoverage, self).__init__()

    # Check the rangke of k.
    if k < 1:
      raise ValueError('The argument k is not positive')
    self.k = int(k)

  def covered(self, internals, **kwargs):
    '''Returns a list of top-k neuron coverage vectors.
    
    Args:
      internals: A list of the values of internal neurons in each layer.
      kwargs: Not used. Present for the compatibility with the super class.
    
    Returns:
      A top-k neuron coverage vecter that identifies which neurons within
      highest k-th values in their layers.

    Example:

    >>> from adapt.metric import TKNC
    >>> import tensorflow as tf
    >>> metric = TKNC(1)
    >>> internals = [tf.random.normal((3,)), tf.random.normal((2,)), tf.random.normal((3,))]
    >>> for x in internals:
    ...   print(x)
    ...
    tf.Tensor([-0.07854115 -0.6883012  -0.8056681 ], shape=(3,), dtype=float32)
    tf.Tensor([-2.316517  -0.2972477], shape=(2,), dtype=float32)
    tf.Tensor([-0.6506158 -0.2905271  1.0730451], shape=(3,), dtype=float32)
    >>> covered = metric(internals=internals)
    >>> for x in covered:
    ...   print(x)
    ...
    [ True False False]
    [False  True]
    [False False  True]
    '''

    # A list to store top-k neuron coverage vectors.
    covered = []

    # Loop for each layer.
    for i in internals:

      # Guard for the value of k.
      k = min(self.k, i.shape.as_list()[0])

      # Find out the indices of k highest values.
      idx = np.argpartition(i, -k)[-k:]

      # Create a top-k coverage vector.
      vec = np.zeros(i.shape, dtype=bool)
      vec[idx] = True
      covered.append(vec)

    return np.array(covered)


  def __repr__(self):
    '''Returns a string representation of object.
    
    Example:
    
    >>> from adapt.metric import TKNC
    >>> metric = TKNC()
    >>> metric
    TopkNeuronCoverage(k=0.5)
    '''

    return 'TopkNeuronCoverage(k={})'.format(self.k)
