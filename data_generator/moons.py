import numpy as np
from matplotlib import pyplot as plt

def moons():

    n_samples = 100
    n_samples_out = n_samples // 2 # 50 observations in one class
    n_samples_in = n_samples - n_samples_out # 50 observations in the other

    rng = np.random.RandomState(132)

    # data set not linearly separable
    outer_circle_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circle_y = np.sin(np.linspace(0, np.pi, n_samples_out))

    inner_circle_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circle_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack((np.append(outer_circle_x, inner_circle_x),
                  np.append(outer_circle_y, inner_circle_y))).T

    y = np.hstack([np.zeros(n_samples_in, dtype=np.intp),
                   np.ones(n_samples_out, dtype=np.intp)])


    # plot two moons without noise
    fig1 = plt.figure(1)
    plt.plot(X[[i for i,j in zip(range(0, y.shape[0]), y) if j == 0], 0],
             X[[i for i,j in zip(range(0, y.shape[0]), y) if j == 0], 1], 'ob')
    plt.plot(X[np.where(y > 0), 0], X[np.where(y > 0), 1], 'or')
    plt.title("Two moons without noise")
    plt.show()

    mu, sigma = 0, 0.1 # mean and standard deviation

    X += rng.normal(mu, sigma, X.shape)
    # X += np.random.normal(mu, sigma, X.shape)

    # plot two moons with noise
    fig2 = plt.figure(2)
    plt.plot(X[np.where(y < 1), 0], X[np.where(y < 1), 1], 'ob')
    plt.plot(X[np.where(y > 0), 0], X[np.where(y > 0), 1], 'or')
    plt.title("Two moons with noise")
    plt.show()
    return X, y
