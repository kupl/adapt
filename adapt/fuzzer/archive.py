from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from imageio import imwrite
from pathlib import Path
import numpy as np

def Archive(input, label, append='meta'):
  '''Helper function for creating an archiving object.

  Args:
    input: An initial input.
    label: A label that the initial input classified into.
    append: An option that specifies the data that archive stores. Should be one
      of "meta" or "all". By default, "meta" will be used.

  Returns:
    An object created from ArchiveMeta if append is "meta", or an object creaed
    from ArchiveAll if append is "all".

  Raises:
    ValueError: When append is not one of "meta" or "all"
  '''

  # Convert to lower case.
  append = append.lower()

  # Return ArchiveMeta object.
  if append == 'meta':
    return ArchiveMeta(input, label)

  # Return ArchiveAll object.
  elif append == 'all':
    return ArchiveAll(input, label)
  
  # Unknown append option.
  else:
    raise ValueError('The argument append must be one of "meta" or "all".')


class ArchiveBase(ABC):
  '''A class for saving the result of testing (used as an implementation base).
  
  This class will store the inputs, labels of them, and distances of them.
  '''

  def __init__(self, input, label):
    '''Create a archive.
    
    Args:
      input: An initial input.
      label: A label that the initial input classified into.
    '''

    # Save original properties.
    self.input = np.array(input)
    self.label = label

    # Create meta variables.
    self.total = 0
    self.adversarials = 0
    self.count = defaultdict(int)

    self.found_labels = defaultdict(bool)
    self.distance = defaultdict(list)

    # Storage for created inputs.
    self.inputs = defaultdict(list)

  def add(self, input, label, distance):
    '''Add a newly found input.
    
    Args:
      input: A newly found input.
      label: A label that the newly found input classified into.
      distance: A distance (e.g. l2 distance) from origianl input.
    '''

    # Update meta varaibles.
    self.total += 1
    if label != self.label:
      self.adversarials += 1
    self.count[label] += 1

    self.found_labels[label] = True
    self.distance[label].append(distance)

    self.append(input, label)

  @abstractmethod
  def append(self, input, label):
    '''Append a created input.

    *** This method should be implemented. ***
    
    Args:
      input: A created input.
      label: A label that the created input classified into.
    '''

  def summary(self, file=None):
    '''Print the summary of the archive.
    
    Args:
      file: A output stream to print. By default, use stdout.
    '''

    print('----------', file=file)

    # Print meta data of total.
    print('Total inputs: {}'.format(self.total), file=file)
    print('  Average distance: {}'.format(np.mean(np.concatenate([self.distance[label] for label in self.distance.keys()]))), file=file)
    
    # Print meta data of adversarials.
    print('Total adversarials: {}'.format(self.adversarials), file=file)
    print('  Average distance: {}'.format('-' if self.adversarials == 0 else np.mean(np.concatenate([self.distance[label] for label in self.distance.keys() if label != self.label]))), file=file)

    print('----------', file=file)

    # Print meta data for original label.
    print('Original label: {}'.format(self.label), file=file)
    print('  Count: {}'.format(self.count[self.label]), file=file)
    print('  Average distance: {}'.format('-' if self.count[self.label] == 0 else np.mean(self.distance[self.label])), file=file)

    # Print meta data for each label found, except for original label.
    for label in self.found_labels.keys():

      # Skip original label.
      if label == self.label:
        continue

      print('----------', file = file)

      # Print meta data for each label found.
      print('Label: {}'.format(label), file=file)
      print('  Count: {}'.format(self.count[label]), file=file)
      print('  Average distance: {}'.format(np.mean(self.distance[label])), file=file)
    
    print('----------', file=file)

  def save_inputs(self, path, deprocess=None, prefix=None, lowest_distance=False):
    '''Save inputs in the archive.

    This method will save inputs in the `path` folder. The file name will be set as
    "{label of a found input}-{identifier number}" with the `prefix` in front of it.

    Args:
      path: A folder to save inputs.
      deprocess: deprocess function that applied before saving input. By default,
        use an identity function.
      prefix: A prefix of the file name. By default, "{original label}-" will
        be used.
      lowest_distance: A boolean. If true, find one with the lowest distance,
        and save it.
    '''

    # Create the output folder.
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Check prefix.
    if not prefix:
      prefix = str(self.label) + '-'
    prefix = str(prefix)

    # For each label found.
    for label in self.found_labels.keys():

      # If lowest distance.
      if lowest_distance:
        lowest = np.argmin(self.distance[label])
        inputs = [self.inputs[label][lowest]]

      # Else.
      else:
        inputs = self.inputs[label]

      # Save inputs.
      for i, img in enumerate(inputs):
        imwrite(path / '{}{}-{}'.format(prefix, label, i), deprocess(img))

class ArchiveMeta(ArchiveBase):
  '''An archive class that only stores meta data (label and distance)'''

  def append(self, input, label):
    '''Append a created input.

    Args:
      input: A created input.
      label: A label that the created input classified into.
    '''

    # Do nothing.

class ArchiveAll(ArchiveBase):
  '''An archive class that only stores all data'''

  def append(self, input, label):
    '''Append a created input.

    Args:
      input: A created input.
      label: A label that the created input classified into.
    '''

    # Add the created input.
    self.inputs[label].append(np.array(input))
