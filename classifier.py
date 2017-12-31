import numpy as np
from random import randint
import matplotlib.pyplot as plt


def read_data(file, prediction):
    file_opened = open("./dataset/" + file, "r")
    array = []
    i = 0
    nb_attributes = 0
    for line in file_opened.readlines():
        line_strip = line.strip().split(',')
        if i == 0:
            nb_attributes = len(line_strip) -1
        if len(line_strip) == nb_attributes + 1:
            array.append(line_strip)
        i += 1
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


class Classifier(object):
    def __init__(self, dataset=None):
        self.dataset = dataset
        if dataset is not None:
            self.train_dataset, self.test_dataset = self.split_dataset(proportion=0.75)
            self.nb_explanatory_var = len(self.train_dataset[0]) - 1
            self.nb_sample = len(dataset)
        self.w = [0.0] * self.nb_explanatory_var
        self.w0 = 0

    def learn(self, max_iter, eta, dataset=None, visualization=False):
        if visualization:
            vis = Visualization(3, 9)
        samples = []
        if dataset is None:
            dataset = self.dataset
            nb_sample = self.nb_sample
        else:
            nb_sample = len(dataset)
        for _ in range(max_iter):
            i = randint(0, nb_sample - 1)
            x = dataset[i][1:]
            y = dataset[i][0]
            samples.append(list(dataset[i][1:]))
            self.learn_step(x, y, eta)
            if visualization:
                vis.animate(self.w, self.w0, samples)
        #print("w =", self.w, "w0 =", self.w0)
        return list(self.w), self.w0

    def learn_step(self, x, y, eta):
        pass

    def evaluate(self, test_dataset=None):
        if test_dataset is None:
            test_dataset = self.test_dataset
        nb_correct = 0
        for i in range(len(test_dataset)):
            x = test_dataset[i][1:]
            y = test_dataset[i][0]
            # print(x, y)
            if (y * ((np.dot(self.w, x)) + self.w0)) > 0:
                nb_correct = nb_correct + 1
        result = nb_correct / len(test_dataset)
        #print("% of good classification =", result)
        return result

    def split_dataset(self, proportion):
        dataset = list(self.dataset)
        s = []
        for i in range(int(proportion * len(dataset))):
            index = randint(0, len(dataset) - 1)
            s.append(dataset[index])
            del (dataset[index])
        t = list(dataset)
        return s, t

    def miss(self, dataset=None):
        if dataset is None:
            dataset = self.train_dataset
        miss = []
        for i in range(len(dataset)):
            x = dataset[i][1:]
            y = dataset[i][0]
            miss.append((y * ((np.dot(self.w, x)) + self.w0)) > 0)
        return miss, [x if x == 1 else -1 for x in miss]


class Perceptron(Classifier):
    def __init__(self, dataset=None):
        super().__init__(dataset)

    def learn_step(self, x, y, eta=1):
        if y * ((np.dot(self.w, x)) + self.w0) <= 0:
            self.w0 = self.w0 + eta * y
            self.w = self.w + eta * y * x


class Adaline(Classifier):
    def __init__(self, dataset=None):
        super().__init__(dataset)

    def learn_step(self, x, y, eta=1):
        prediction = ((np.dot(self.w, x)) + self.w0)
        self.w0 = self.w0 + eta * (y - prediction)
        self.w = self.w + eta * (y - prediction) * x


class Visualization(object):
    def __init__(self, x1, x2):
        self.graph_x = np.linspace(x1, x2, 100)
        self.graph_y = np.linspace(0, 0, 100)
        self.plot, = plt.plot(self.graph_x, [0] * 100)
        self.plot_samples = None
        plt.axis([x1, x2, 2, 5])

    def animate(self, w, w0, samples):
        if w[1] != 0:
            for j in range(len(self.graph_x)):
                self.graph_y[j] = (-1 * w0 - 1 * w[0] * self.graph_x[j]) / w[1]
        self.plot.set_ydata(self.graph_y)
        # self.plot_samples.set_xdata(np.array(samples)[:, 0])
        # self.plot_samples.set_ydata(np.array(samples)[:, 1])
        self.plot_samples = plt.plot(np.array(samples)[:, 0], np.array(samples)[:, 1], 'o')
        plt.pause(0.02)


    @staticmethod
    def show():
        plt.show()


class CrossValidation(object):
    # k-fold cross-validation
    # Cross-validation on perceptron is non-sense since learning rate has no impact on perceptron.
    # Cross-validation on adaline returns optimal learning rate,
    # which is dependant on the number of learning iterations max_iter.
    # That's legit since gradient descent with low learning-rate is slower (need more iterations)
    # but is more accurate when it is not trapped by local extrema.
    def __init__(self, dataset, k, eta_range, max_iter):
        self.dataset = dataset
        self.k = k
        self.max_iter = max_iter
        self.eta_range = eta_range

    def cross_validation(self):
        epsilon = []
        for eta in self.eta_range:
            epsilon_eta_array = []
            for i in range(self.k):
                adaline = Adaline(self.dataset)
                vk, ek = self.split_validate_train(self.k, i)
                adaline.learn(max_iter=self.max_iter, eta=eta, dataset=ek)
                epsilon_eta_array.append(adaline.evaluate(test_dataset=vk))
            epsilon_eta = np.mean(epsilon_eta_array)
            epsilon.append(epsilon_eta)
        ind_max = np.argmax(epsilon)
        print("found optimal eta =", self.eta_range[ind_max])
        return self.eta_range[ind_max]

    def split_validate_train(self, k, i):
        dataset = self.dataset
        n = len(dataset)
        validate_ind1 = i * round(n / k)
        validate_ind2 = min(i * round(n / k) + k, n)
        validate = dataset[validate_ind1:validate_ind2]
        train = dataset[0:validate_ind1] + dataset[validate_ind2:n]
        return validate, train


class AdaBoost(Classifier):
    def __init__(self, dataset):
        super().__init__(dataset)

    def learn(self, max_iter, eta, dataset=None, visualization=False):
        for m in range(max_iter):
            adaline = Adaline()
            w, w0 = adaline.learn(max_iter=2000, eta=eta, dataset=self.train_dataset)
            #print(adaline.evaluate())
            miss, miss2 = adaline.miss(dataset=self.train_dataset)
            err_m = np.dot(weights, miss) / sum(w)


if __name__ == '__main__':
    dataset = read_data(file="ionosphere.data", prediction="b")
    #print(dataset)
    #dataset = list(np.array(dataset)[:, 0:3])

    # perceptron = Perceptron(dataset)
    # perceptron.learn(max_iter=2000, eta=0.1)
    # perceptron.evaluate()
    # adaline = Adaline(dataset)
    # adaline.learn(max_iter=3000, eta=0.002, visualization=False)
    # adaline.evaluate()
    #
    # eta_range = [0.00001, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    # cross_validation = CrossValidation(dataset=dataset, k=5, eta_range=eta_range, max_iter=3000)
    # cross_validation.cross_validation()

    adaboost = AdaBoost(dataset)
    adaboost.learn(max_iter=10, eta=0.001)
