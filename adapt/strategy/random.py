import numpy as np

from adapt.strategy.strategy import Strategy

class RandomStrategy(Strategy):
  '''A strategy that randomly selects neurons from all neurons.
  
  This strategy selects neurons from a set of all neurons in the network,
  except for the neurons that located in skippable layers.
  '''

  def select(self, k):
    '''Seleck k neurons randomly.

    Select k neurons randomly from a set of all neurons in the network,
    except for the neurons that located in skippable layers.

    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of location of the selected neurons.
    '''

    # Choose k neurons and return their location.
    indices = np.random.choice(len(self.neurons), size=k, replace=False)
    return [self.neurons[i] for i in indices]
