#################################
# Your name: Tom Jacob
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
import scipy as sp
from scipy import special

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""
# %%
"""
:returns train_data, train_labels, validation_data, validation_labels, test_data, test_labels
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
# %%
"""
returns the best classifier for C, eta_0
"""


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    ℓ(w, x, y) = C · max{0, 1 −y⟨w, x⟩} + 0.5∥w∥^2
    """
    classifier = np.zeros(784)

    for iteration in range(1, T + 1):
        i = np.random.randint(0, len(data), 1)  # sample data uniformly
        classifier = classifier.reshape(784, )  # prevents bugs in np for some reason

        condition = (np.dot(data[i], classifier)) * labels[i]  # calc  y⟨w, x⟩
        weight = eta_0 / iteration
        if condition < 1:
            # update classifier: wt+1 = (1 − ηt)wt + ηtCyixi
            tmp = np.multiply(1 - weight, classifier)
            classifier = np.add(tmp, np.multiply(weight * C * labels[i], data[i]))
        else:
            classifier = np.multiply(1 - weight, classifier)  # update classifier: w_t+1 = (1-eta_t) * w_t

    return classifier


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    ℓlog (w, x, y) = log(1+exp(-xyw))
    """
    classifier = np.zeros(784)
    t_axis = []
    norm_axis = []

    for iteration in range(1, T + 1):
        i = np.random.randint(0, len(data) - 1)  # sample data uniformly
        eta_t = eta_0 / iteration

        power = -labels[i] * np.dot(classifier, data[i])  # calc  y⟨w, x⟩
        softmax = sp.special.softmax(np.array([0, power]))[1]

        a = -labels[i] * softmax * data[i]
        classifier -= eta_t * a


        t_axis.append(iteration)
        norm_axis.append(sp.linalg.norm(classifier))

    plt.plot(t_axis, norm_axis)
    plt.xscale('log')
    plt.xlabel('iteration')
    plt.ylabel("norm")
    plt.show()

    return classifier


# %%
def question1_a(train_data, train_labels, validation_data, validation_labels):
    x_axis = [1.1 ** (i - 5) for i in range(11)]
    accuracy_axis = np.zeros(11)
    for i in range(11):
        classifier = SGD_hinge(train_data, train_labels, 1, x_axis[i], 1000)
        accuracy_axis[i] = calc_accuracy(classifier, validation_data, validation_labels)

    plt.plot(x_axis, accuracy_axis)
    plt.xlabel("η_0")
    plt.ylabel("Average accuracy")
    plt.xscale('log')
    plt.title("Question 1a")
    plt.show()
    print(np.argmax(accuracy_axis) - 5)


def questio1_b(train_data, train_labels, validation_data, validation_labels):
    x_axis = [15 ** (i - 5) for i in range(11)]
    accuracy_axis = np.zeros(11)
    for i in range(11):
        classifier = SGD_hinge(train_data, train_labels, x_axis[i], 0.9, 1000)
        accuracy_axis[i] = calc_accuracy(classifier, validation_data, validation_labels)

    plt.plot(x_axis, accuracy_axis)
    plt.xlabel("C")
    plt.ylabel("Average accuracy")
    plt.xscale('log')
    plt.title("Question 1b")
    plt.show()
    print((np.argmax(accuracy_axis) - 5))


def question1_c(train_data, train_labels):
    classifier = SGD_hinge(train_data, train_labels, 15 ** -3, 0.9, 20000)
    plt.imshow(np.reshape(classifier, (28, 28)), interpolation='nearest')
    plt.show()


def question1_d(train_data, train_labels, validation_data, validation_labels):
    classifier = SGD_hinge(train_data, train_labels, 15 ** -3, 0.9, 20000)
    return calc_accuracy(classifier, validation_data, validation_labels)


# %%
def question2_a():

    eta_res = []
    acc_res = []

    for power in range(-5, 6):
        eta_0 = 10 ** power
        acc10 = np.zeros(10)

        for i in range(len(acc10)):
            classifier = SGD_log(train_data, train_labels, eta_0, 1000)
            acc10[i] = calc_accuracy(classifier, validation_data, validation_labels)

        eta_res.append(eta_0)
        acc_res.append(np.average(acc10))

    plt.plot(eta_res, acc_res)
    plt.xlabel('eta')
    plt.ylabel("accuracy")
    plt.xscale('log')
    plt.show()

    best_eta = eta_res[np.argmax(acc_res)]
    return best_eta

def question2_b():
    classifier = SGD_log(train_data, train_labels, 10 ** -5, 20000)
    print(calc_accuracy(classifier, validation_data, validation_labels))
    plt.imshow(np.reshape(classifier, (28, 28)), interpolation='nearest')
    plt.show()

def question2_c():
    classifier = SGD_log(train_data, train_labels, 10 ** -5, 20000)
    return sp.linalg.norm(classifier)

#%%
#################################
def calc_accuracy(classifier, validation_data, validation_labels):
    out: int = 0
    for index in range(len(validation_data)):
        if np.dot(classifier, validation_data[index]) >= 0:
            label = 1
        else:
            label = -1
        out += 0 if label == validation_labels[index] else 1

    return 1 - (out / len(validation_data))
#################################
# %%
# question1_a(train_data, train_labels, validation_data, validation_labels)
# question1_b(train_data, train_labels, validation_data, validation_labels)
# question1_c(train_data, train_labels)
# acc = question1_d(train_data, train_labels,test_data, test_labels)
# print(acc)

question2_c()
# question2_b()