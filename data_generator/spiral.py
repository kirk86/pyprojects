import numpy as np
from matplotlib import pyplot as plt
# import pdb
def spiral():

  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros( (N * K, D) ) # data matrix (each row = single example)
  y = np.zeros(N * K, dtype='uint8') # class labels
  idx = range(0, 400, 100)
  # a = np.array([])
  # b = np.array([])
  for j in xrange(K):
    radius = np.linspace(0, 1, N) # radius
    theta  = np.linspace( j * 4, (j + 1) * 4, N ) + np.random.randn(N) * 0.2 # theta
    # pdb.set_trace()
    X[ idx[j]:idx[j + 1] ] = np.vstack((radius * np.sin(theta), radius *
                                        np.cos(theta))).T
    y[ idx[j]:idx[j + 1] ] = j
  # lets visualize the data:
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.show()
  return X, y
