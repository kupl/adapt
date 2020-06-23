from itertools import cycle
import numpy as np

from adapt.strategy.strategy import Strategy

class DLFuzzRoundRobin(Strategy):
  '''A round-robin strategy that cycles 3 strategies that suggested by DLFuzz.
  
  DLFuzz suggest 4 different strategy as follows:
  * Select neurons that are most covered.
  * Select neurons that are rarely covered.
  * Select neurons with the largest weights.
  * Select neurons that have values near threshold.
  From the suggested strategies, 4th strategy is highly subordinate to the
  neuron coverage. Therefore, 4th strategy is not included in round-robin
  strategy. Please, see the following paper for more details:

  DLFuzz: Differential Fuzzing Testing of Deep Learning Systems
  https://arxiv.org/abs/1808.09413
  '''

  def __init__(self, network, weight_portion=0.1, order=None):
    '''Create a DLFuzz round-robin strategy.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.
      weight_portion: A portion of neurons to use for 3rd strategy.
      order: The order of round-robin. By default, [1, 2, 3].

    Raises:
      ValueError: When weight_portion is not in [0, 1].

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import DLFuzzRoundRobin
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = DLFuzzRoundRobin(network)
    '''

    super(DLFuzzRoundRobin, self).__init__(network)

    # A vector that stores how many times each neuron is covered.
    self.covered_count = None

    # A pool of weights.
    weights = []

    # Collect all weights.
    for l in network.layers[:-1]:

      # Get weights.
      w = l.get_weights()

      if len(w) > 0:
        #  Use only weights, not biases.
        w = w[0]

      # If layer without weights (e.g. Pooling layer).
      else:
        w = np.zeros(l.output.shape[1:])
      
      # Calculate the weight of the neurons.
      for ni in range(l.output.shape[-1]):
        weights.append(np.mean(w[..., ni]))

    # Guard for the range of weight portion
    if weight_portion < 0 or weight_portion > 1:
      raise ValueError('The argument weight_portion is not in [0, 1].')
    self.weight_portion = weight_portion
    
    # Find the neurons with high values.
    k = int(len(self.neurons) * self.weight_portion)
    self.weight_indices = np.argpartition(weights, -k)[-k:]

    # Round-robin cycle
    if not order:
      order = [1, 2, 3]
    self.order = cycle(order)

    # Start from the first strategy in the order.
    self.current = next(self.order)

  def select(self, k):
    '''Select k neurons with the current strategy.
    
    Seleck k neurons, and returns their location.

    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.

    Raises:
      ValueError: When the current strategy is unknown.
    '''

    # First strategy.
    if self.current == 1:
      
      # Find k most covered neurons.
      indices = np.argpartition(self.covered_count, -k)[-k:]

    # Second strategy.
    elif self.current == 2:
      
      # Find k rarest covered neurons.
      indices = np.argpartition(self.covered_count, k - 1)[:k]

    # Third strategy.
    elif self.current == 3:
      
      # Randomly samples from the neurons with high weights.
      indices = np.random.choice(self.weight_indices, size=k, replace=False)
    
    # Unknown.
    else:
      raise ValueError('Unknown strategy. The strategy must be 1, 2, or 3.')

    return [self.neurons[i] for i in indices]

  def init(self, covered, **kwargs):
    '''Initialize the variable of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.

    Raises:
      ValueError: When the size of the passed coverage vectors are not matches
        to the network setting.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)
    if len(covered) != len(self.neurons):
      raise ValueError('The number of neurons in network does not matches to the setting.')

    # Initialize the number of covering for each neuron.
    self.covered_count = np.zeros_like(covered, dtype=int)
    self.covered_count += covered

    return self

  def update(self, covered, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''
    
    # Flatten coverage vectors.
    covered = np.concatenate(covered)

    # Update the number of covering for each neuron.
    self.covered_count += covered

  def next(self):
    '''Move to the next strategy.

    Returns:
      Self for possible call chains.
    '''
    
    # Get the next strategy.
    self.current = next(self.order)

    return self

class MostCoveredStrategy(DLFuzzRoundRobin):
  '''A strategy selects most covered
  
  This strategy selects most covered neurons. This strategy is first introduced
  in the following paper. Please, see the following paper for more details:

  DLFuzz: Differential Fuzzing Testing of Deep Learning Systems
  https://arxiv.org/abs/1808.09413

  Since this strategy is part of DLFuzzRoundRobin, this implementation re-use
  the implementation of DLFuzzRoundRobin.
  '''

  def __init__(self, model):
    '''Create a strategy.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import MostCoveredStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = MostCoveredStrategy(network)
    '''

    # Re-use the implementation of super class.
    super(MostCoveredStrategy, self).__init__(model)

    # Set current strategy as 1.
    self.current = 1

    # Remove unnecessary variables.
    del self.weight_portion
    del self.weight_indices
    del self.order

  def next(self):
    '''Do nothing.
    
    Returns:
      Self for possible call chains.
    '''

    return self
