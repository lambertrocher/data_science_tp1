import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from time import sleep


def read_data(file, nb_attributes, prediction):
    file_opened = open(file, "r")
    array = []
    for line in file_opened.readlines():
        line_strip = line.strip().split(',')
        if len(line_strip) == nb_attributes + 1:
            array.append(line_strip)
    for line in array:
        if len(line) == nb_attributes + 1:
            if line[nb_attributes] == prediction:
                line.insert(0, 1.0)
            else:
                line.insert(0, -1.0)
            del (line[nb_attributes + 1])
            for i in range(nb_attributes):
                line[i + 1] = float(line[i + 1])
    for i in range(len(array)):
        array[i] = np.array(array[i])
    return array


def split_base_into_learn_and_test(base, proportion):
    s = []
    for i in range(int(proportion * len(base))):
        index = randint(0, len(base) - 1)
        s.append(base[index])
        del (base[index])
    t = list(base)
    return s, t


def split_base_into_validate_and_train(base, k, i):
    n = len(base)
    validate_ind1 = i * round(n / k)
    validate_ind2 = min(i * round(n / k) + k, n)
    validate = base[validate_ind1:validate_ind2]
    train = base[0:validate_ind1] + base[validate_ind2:n]
    return validate, train


def train_perceptron(base, max_iter, eta, eta_2, tt):
    considered_examples = []
    nb_attributes = len(base[0]) - 1
    w0 = 0.0
    w = [0.0] * nb_attributes

    w0_2 = 0.0
    w_2 = [0.0] * nb_attributes


    m = len(base)
    for _ in range(max_iter):

        i = randint(0, m - 1)
        x = base[i][1:]
        y = base[i][0]


        considered_examples.append(list(base[i][1:]))

        plt.show()
        # sleep(1)
        # plt.close('all')
        if y * ((np.dot(w, x)) + w0) <= 0:
            w0 = w0 + eta * y
            w = w + eta * y * x

            w0_2 = w0_2 + eta_2 * y
            w_2 = w_2 + eta_2 * y * x
    # print("% good classification", compute_error(tt,w0,w))

        print("x", x, "y", y)
        print("w0", w0, "w", w)
        print("w0_2", w0_2, "w_2", w_2)

        print(compute_error(tt, w0, w))
        graph_x = np.linspace(-2, 10, 30)
        graph_y = np.linspace(0, 0, 30)
        if w[1] != 0:
            for j in range(len(graph_x)):
                graph_y[j] = (-1 * w0 - 1 * w[0] * graph_x[j]) / w[1]

        plt.axis([-2, 10, -2, 10])
        plt.plot(np.array(considered_examples)[:, 0], np.array(considered_examples)[:, 1], 'o')
        plt.plot(graph_x, graph_y)

        print(compute_error(tt, w0_2, w_2))
        graph_x_2 = np.linspace(-2, 10, 45)
        graph_y_2 = np.linspace(0, 0, 45)
        if w_2[1] != 0:
            for j in range(len(graph_x_2)):
                graph_y_2[j] = (-1 * w0_2 - 1 * w_2[0] * graph_x_2[j]) / w_2[1] + 0.1

        plt.axis([-2, 10, -2, 10])
        plt.plot(graph_x_2, graph_y_2)

    return w0, w


def compute_error(base_validation, w0, w):
    nb_correct = 0
    for i in range(len(base_validation)):
        x = base_validation[i][1:]
        y = base_validation[i][0]
        #print(x, y)
        if (y * ((np.dot(w, x)) + w0)) > 0:
            nb_correct = nb_correct + 1
    return nb_correct / len(base_validation)


# k-fold cross-validation
def cross_validation(st, tt, k):
    eta_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    epsilon = []
    for eta in eta_range:
        epsilon_eta_array = []
        for i in range(k):
            vk, ek = split_base_into_validate_and_train(st, k, i)
            w0, w = train_perceptron(base=ek, max_iter=10, eta=eta, tt=ek)
            epsilon_eta_array.append(compute_error(base_validation=vk, w0=w0, w=w))
        epsilon_eta = np.mean(epsilon_eta_array)
        epsilon.append(epsilon_eta)
    ind_max = np.argmax(epsilon)
    return eta_range[ind_max]


def main():
    base = read_data(file="iris.data", nb_attributes=4, prediction="Iris-setosa")
    base = list(np.array(base)[:, 0:3])

    print(base)

    st, tt = split_base_into_learn_and_test(base=base, proportion=0.1)

    print("taille apprentissage", len(st))

    w0, w = train_perceptron(base=st, max_iter=200, eta=0.01, eta_2=1, tt=tt)
    print("w0 =", w0, "w =", w)
    print("% good classification =", compute_error(base_validation=tt, w0=w0, w=w))

    # for eta in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    #     w0, w = train_perceptron(base=st, max_iter=30000, eta=eta, tt=tt)
    #     print("w0 =", w0, "w =", w)
    #     print("% good classification =", compute_error(base_validation=tt,w0=w0,w=w))
    #
    # for _ in range(10):
    #     best_eta = cross_validation(st, tt, 5)
    #     print(best_eta)


main()
