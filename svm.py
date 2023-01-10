import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    assert isinstance(yy, object)
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def claus_a(X_train, y_train):
    linear_kernel = svm.SVC(C=10, kernel='linear')
    linear_kernel.fit(X_train, y_train)
    quadratic_kernel = svm.SVC(C=10, kernel='poly', degree=2)
    quadratic_kernel.fit(X_train, y_train)
    cubic_kernel = svm.SVC(C=10, kernel='poly', degree=3)
    cubic_kernel.fit(X_train, y_train)

    models = np.stack((linear_kernel, quadratic_kernel, cubic_kernel))
    titles = ['linear', 'polynomial of degree 2', 'polynomial of degree 3']
    plot_results(models, titles, X_train, y_train, plot_sv=False)


def claus_b(X_train, y_train):
    linear_kernel = svm.SVC(C=10, kernel='linear', coef0=1)
    linear_kernel.fit(X_train, y_train)
    quadratic_kernel = svm.SVC(C=10, kernel='poly', degree=2, coef0=1)
    quadratic_kernel.fit(X_train, y_train)
    cubic_kernel = svm.SVC(C=10, kernel='poly', degree=3)
    cubic_kernel.fit(X_train, y_train)

    models = np.stack((linear_kernel, quadratic_kernel, cubic_kernel))
    titles = ['linear', 'non-homogen` deg 2', 'non-homogen` deg 3']
    plot_results(models, titles, X_train, y_train, plot_sv=False)


def claus_c(X_train, y_train):
    quadratic_kernel = svm.SVC(C=10, kernel='poly', degree=2)
    quadratic_kernel.fit(X_train, y_train)
    rbf_small_kernel = svm.SVC(C=10, kernel='rbf', gamma=3)
    rbf_small_kernel.fit(X_train, y_train)
    rbf_mid_kernel = svm.SVC(C=10, kernel='rbf', gamma=10)
    rbf_mid_kernel.fit(X_train, y_train)
    rbf_big_kernel = svm.SVC(C=10, kernel='rbf', gamma=100)
    rbf_big_kernel.fit(X_train, y_train)

    models_1 = np.stack((quadratic_kernel, rbf_small_kernel))
    models_2 = np.stack((rbf_mid_kernel, rbf_big_kernel))
    titles_1 = ['polynomial deg 2', 'rbf kernel gamma=3']
    titles_2 = ['rbf kernel gamma=10', 'rbf kernel gamma=100']
    plot_results(models_1, titles_1, X_train, y_train, plot_sv=False)
    plot_results(models_2, titles_2, X_train, y_train, plot_sv=False)


C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100

# Data is labeled by a circle

radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
angles = 2 * math.pi * np.random.random(2 * n)
X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
X2 = (radius * np.sin(angles)).reshape((2 * n, 1))

X = np.concatenate([X1, X2], axis=1)
y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])

# claus_a(X, y)

# claus_b(X, y)

negative_labels = -np.ones((n, 1))
for i in range(n):
    negative_labels[i] = np.random.choice((1, -1), size=1, p=[0.1, 0.9])

y_c = np.concatenate([np.ones((n, 1)), negative_labels]).reshape([-1])

claus_c(X, y_c)
