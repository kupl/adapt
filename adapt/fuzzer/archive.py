from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from imageio import imwrite
from pathlib import Path
import numpy as np

def Archive(image, label, append='meta'):
  '''Helper function for creating an archiving object.

  Args:
    image: An initial image.
    label: A label that the initial image classified into.
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
    return ArchiveMeta(image, label)

  # Return ArchiveAll object.
  elif append == 'all':
    return ArchiveAll(image, label)
  
  # Unknown append option.
  else:
    raise ValueError('The argument append must be one of "meta" or "all".')


class ArchiveBase(ABC):
  '''A class for saving the result of testing (used as an implementation base).
  
  This class will store the images, labels of them, and distances of them.
  '''

  def __init__(self, image, label):
    '''Create a archive.
    
    Args:
      image: An initial image.
      label: A label that the initial image classified into.
    '''

    # Save original properties.
    self.image = np.array(image)
    self.label = label

    # Create meta variables.
    self.total = 0
    self.adversarials = 0
    self.count = defaultdict(int)

    self.found_labels = defaultdict(bool)
    self.distance = defaultdict(list)

    # Storage for created images.
    self.images = defaultdict(list)

  def add(self, image, label, distance):
    '''Add a newly found image.
    
    Args:
      image: A newly found image.
      label: A label that the newly found image classified into.
      distance: A distance (e.g. l2 distance) from origianl image.
    '''

    # Update meta varaibles.
    self.total += 1
    if label != self.label:
      self.adversarials += 1
    self.count[label] += 1

    self.found_labels[label] = True
    self.distance[label].append(distance)

    self.append(image, label)

  @abstractmethod
  def append(self, image, label):
    '''Append a created image.

    *** This method should be implemented. ***
    
    Args:
      image: A created image.
      label: A label that the created image classified into.
    '''

  def summary(self, file=None):
    '''Print the summary of the archive.
    
    Args:
      file: A output stream to print. By default, use stdout.
    '''

    print('----------', file=file)

    # Print meta data of total.
    print('Total images: {}'.format(self.total), file=file)
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

  def save_images(self, path, deprocess=None, prefix=None, lowest_distance=False):
    '''Save images in the archive.

    This method will save images in the `path` folder. The file name will be set as
    "{label of a found image}-{identifier number}" with the `prefix` in front of it.

    Args:
      path: A folder to save images.
      deprocess: deprocess function that applied before saving image. By default,
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
        images = [self.images[label][lowest]]

      # Else.
      else:
        images = self.images[label]

      # Save images.
      for i, img in enumerate(images):
        imwrite(path / '{}{}-{}'.format(prefix, label, i), deprocess(img))

class ArchiveMeta(ArchiveBase):
  '''An archive class that only stores meta data (label and distance)'''

  def append(self, image, label):
    '''Append a created image.

    Args:
      image: A created image.
      label: A label that the created image classified into.
    '''

    # Do nothing.

class ArchiveAll(ArchiveBase):
  '''An archive class that only stores all data'''

  def append(self, image, label):
    '''Append a created image.

    Args:
      image: A created image.
      label: A label that the created image classified into.
    '''

    # Add the created image.
    self.images[label].append(np.array(image))
