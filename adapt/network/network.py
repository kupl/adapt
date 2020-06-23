from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras.backend as K

class Network:
  '''A wrapper class for Keras model.
  
  This class will help you get the values from the internal neurons. All models
  used in ADAPT should be wrapped with this class.
  '''

  def __init__(self, model, skippable=None):
    '''Create a Keras model wrapper class from a Keras model.

    Args:
      model: A Keras model. This argument is required.
      skippable: A list of Keras layer classes that can be skipped while getting
        the values. By default, all layers that created from `tensorflow.keras.layers.Flatten`
        and `tensorflow.keras.layers.InputLayer` will be skipped.

    Example:

    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> from adapt import Network
    >>> model = VGG19()
    >>> network = Network(model)
    '''

    self.model = model

    # If skippable is not specified, use default skippable layers.
    if not skippable:
      skippable = [InputLayer, Flatten]
    self.skippable = skippable

    # Functors that returns the outputs of the not skippable layers.
    self.functors = Model(inputs = self.model.input, outputs = [l.output for l in self.model.layers if type(l) not in self.skippable])
    
  def predict(self, x):
    '''Calculate the internal values and the logits of the input.
    
    Args:
      x: An input to process. Currently, Network class does not support batch
        processing. Therefore, the first dimension of the input must be 1.

    Returns:
      A tuple of a list of the values of internal neurons in each layer and logits

    Example:
    
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> from adapt import Network
    >>> import numpy as np
    >>> model = VGG19()
    >>> network = Network(model)
    >>> x = np.random.randn(1, 224, 224, 3)
    >>> x.shape
    (1, 224, 224, 3)
    >>> internal, logits = network.predict(x)
    >>> len(internal)
    23
    >>> logits.shape
    TensorShape([1000])
    '''

    # Get output and normalize to get the output of the neurons.
    outs = [K.mean(K.reshape(l, (-1, l.shape[-1])), axis = 0) for l in self.functors(x)]

    # Return internal outputs and logits.
    internals = outs[:-1]
    logits = outs[-1]
    return internals, logits

  @property
  def layers(self):
    '''A list of layers that is not skippable.
    
    Example:
    
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> from adapt import Network
    >>> model = VGG19()
    >>> network = Network(model)
    >>> len(network.layers)
    24
    '''

    # Return a list of layers which are not skippable.
    return [l for l in self.model.layers if type(l) not in self.skippable]
