import numpy as np
from matplotlib import pyplot as plt

def classifier_surface(X, y, params):

    # plot the resulting classifier
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    X_grid_points = np.vstack((xx.ravel(), yy.ravel())).T
    if len(params) > 2:
        Z = np.dot(np.maximum(0, np.dot(X_grid_points, params[0]) + params[1]), params[2]) + params[3]
    else:
        Z = np.dot(X_grid_points, params[0]) + params[1]
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
    #fig.savefig('spiral_linear.png')
