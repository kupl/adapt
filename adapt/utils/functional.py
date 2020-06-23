import numpy as np

def greedy_max_set(covereds, n=None):
  '''Returns a maximum coverage vector composed of n elements, and index of elements.

  This function solves maximum coverage problem with greedy approach.

  Args:
    covereds: A list of coverage vectors.
    n: Maximum number of elements to use for finding maximum coverage. By default,
      use the length of the `covereds`.
  '''

  # Check arguments.
  covereds = np.array(covereds)
  if n is None:
    n = len(covereds)
  
  # Initialize result variables.
  idxs = []
  max_set = np.zeros_like(covereds[0], dtype = bool)

  # Find n elements.
  for _ in range(n):

    # If there is no room for improvement.
    if np.sum(covereds) == 0:
      break

    # Find next greedy element.
    sums = [np.sum(m) for m in covereds]
    idx = np.argmax(sums)
    chosen = covereds[idx]

    # Update maximum coverage vector.
    max_set = np.bitwise_or(max_set, chosen)
    idxs.append(idx)
    
    # Update candidates.
    chosen = np.bitwise_not(chosen)
    covereds = [np.bitwise_and(m, chosen) for m in covereds]
    
  return max_set, idxs

def coverage(covered):
  '''Calculate a coverage as a floating point number.

  Args:
    covered: A list coverage vectors.

  Returns:
    A coverage as a floating point number.
  '''

  return np.mean(np.concatenate(covered))
