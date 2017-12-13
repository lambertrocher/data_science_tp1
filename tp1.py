from random import randint
import numpy as np


def read_data(file):
    file_iris = open(file, "r")
    iris_array = []
    for line in file_iris.readlines():
        line_strip = line.strip().split(',')
        if len(line_strip) == 5:
            iris_array.append(line_strip)
    for line in iris_array:
        if len(line) == 5:
            if line[4] == 'Iris-setosa':
                line.insert(0, 1.0)
            else:
                line.insert(0, -1.0)
            del(line[5])
            for i in range(4):
                line[i+1] = float(line[i+1])
    for i in range(len(iris_array)):
        iris_array[i] = np.array(iris_array[i])
    return iris_array


def split_base_into_learn_and_test(base, proportion):
    s = []
    for i in range(int(proportion*len(base))):
        index = randint(0, len(base)-1)
        s.append(base[index])
        del(base[index])
    t = list(base)
    return s, t


def split_base_into_validate_and_train(base, k, i):
    n = len(base)
    validate_ind1 = i*round(n/k)
    validate_ind2 = min(i*round(n/k)+k, n)
    validate = base[validate_ind1:validate_ind2]
    train = base[0:validate_ind1]+base[validate_ind2:n]
    return validate, train


def train_perceptron(base, max_iter, eta):
    w0 = 0.0
    w = [0.0] * 4
    m = len(base)
    for t in range(max_iter):
        for i in range(m):
            x = base[i][1:]
            y = base[i][0]
            if y*((np.dot(w, x))+w0) <= 0:
                w0 = w0 + eta*y
                w = w + eta*y*x
    return w0, w


#k-fold cross-validation
def cross_validation(st,tt,k):
    eta_range = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    for eta in eta_range:
        for i in range(k):
            vk, ek = split_base_into_validate_and_train(st, k, i)
            train_perceptron(base=ek,max_iter=10,eta=eta)


def main():
    iris = read_data(file="iris.data")
    st, tt = split_base_into_learn_and_test(base=iris, proportion=0.75)
    w0, w = train_perceptron(base=st, max_iter=10, eta=0.1)
    print(w0, w)

main()

# now we need to choose eta with cross validation
