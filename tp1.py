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
    print(eta)
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


def compute_error(base_validation, w0, w):
    nb_correct = 0
    for i in range(len(base_validation)):
        x = base_validation[i][1:]
        y = base_validation[i][0]
        if (y*((np.dot(w, x))+w0))>0:
            nb_correct = nb_correct+1
    return nb_correct/len(base_validation)


# k-fold cross-validation
def cross_validation(st,tt,k):
    eta_range = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 0.99, 1]
    epsilon = []
    for eta in eta_range:
        epsilon_eta_array = []
        print("eta",eta)
        for i in range(k):
            vk, ek = split_base_into_validate_and_train(st, k, i)
            w0, w = train_perceptron(base=ek, max_iter=10, eta=eta)
            epsilon_eta_array.append(compute_error(base_validation=vk, w0=w0, w=w))
        epsilon_eta = np.mean(epsilon_eta_array)
        epsilon.append(epsilon_eta)


def main():
    iris = read_data(file="iris.data")
    # print(len(iris))
    st, tt = split_base_into_learn_and_test(base=iris, proportion=0.05)
    for eta in [0.00,0.01,0.1,1]:
        w0, w = train_perceptron(base=st, max_iter=1, eta=eta)
        print(w0, w)
        print(compute_error(base_validation=tt,w0=w0,w=w))
    # cross_validation(st, tt, 5)


main()

# now we need to choose eta with cross validation

