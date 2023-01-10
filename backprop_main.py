import backprop_data
import backprop_network
import matplotlib.pyplot as plt
import numpy as np


def q_a():
    print("q_a")
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data, q_='a')


def q_b():
    print("q_b")
    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)
    rates = [0.001, 0.01, 0.1, 1, 10, 100]
    epochs_num = 30

    train_accuracy_per_rate = []
    train_loss_per_rate = []
    test_accuracy_per_rate = []

    for rate in rates:
        print(f'\n______Start learning for rate {rate}______')
        net = backprop_network.Network([784, 40, 10])
        curr_rate_train_accuracy, curr_rate_train_loss, curr_rate_test_accuracy = \
            net.SGD(training_data, epochs=epochs_num, mini_batch_size=10, learning_rate=rate, test_data=test_data,
                    q_='b')

        train_accuracy_per_rate.append(curr_rate_train_accuracy)
        train_loss_per_rate.append(curr_rate_train_loss)
        test_accuracy_per_rate.append(curr_rate_test_accuracy)

        print(f'______Finished learning for rate {rate}______\n')

    zero_to_epoches = np.arange(epochs_num)
    # train_accuracy_per_rate
    for train_acc, rate in zip(train_accuracy_per_rate, rates):
        plt.plot(zero_to_epoches, train_acc, label=f"rate = {rate}")
    plt.xlabel('Epochs')
    plt.ylabel('Train accuracy')
    plt.legend()
    plt.show()

    # train_loss_per_rate
    for train_loss, rate in zip(train_loss_per_rate, rates):
        plt.plot(zero_to_epoches, train_loss, label=f"rate = {rate}")
    plt.xlabel('Epochs')
    plt.ylabel('Train loss')
    plt.legend()
    plt.show()

    # test_accuracy_per_rate
    for test_acc, rate in zip(test_accuracy_per_rate, rates):
        plt.plot(zero_to_epoches, test_acc, label=f"rate = {rate}")
    plt.xlabel('Epochs')
    plt.ylabel('Test accuracy')
    plt.legend()
    plt.show()


def q_c():
    print("q_c")
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data, q_='c')


def q_d():
    print("q_d")
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)
    net = backprop_network.Network([784, 80, 28, 10])
    net.SGD(training_data, epochs=200, mini_batch_size=10, learning_rate=0.01, test_data=test_data, q_='c')


print("Start\n")
# q_a()
# q_b()
# q_c()
q_d()
print("\nFinish")
