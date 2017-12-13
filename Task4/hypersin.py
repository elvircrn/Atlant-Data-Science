import numpy as np
import matplotlib.pyplot as plt

from hyperopt import hp
import hyperopt

class DataSet:
    VALID_SIZE = 10
    TEST_SIZE = 10
    TRAIN_SIZE = 200

    train = []
    test  = []
    validation = []

    @staticmethod
    def get_validation():
        if not DataSet.validation:
            DataSet.validation = sample(DataSet.VALID_SIZE)
        return DataSet.validation
        

    @staticmethod
    def get_test():
        if not DataSet.test:
            DataSet.test = sample(DataSet.TEST_SIZE)
        return DataSet.test


    @staticmethod
    def get_train():
        if not DataSet.train:
            DataSet.train = sample(DataSet.TRAIN_SIZE)
        return DataSet.train


def ground(x):
    return np.sin(x)


def noise(n):
    return (np.random.rand(n) / 200 - 0.025)


def sample(n):
    minx = -32 * np.pi
    maxx = 32 * np.pi

    sample_range = np.abs(maxx - minx)
    sample_points = np.sort(np.random.rand(n)) * sample_range - (sample_range / 2)
    samples = ground(sample_points) - noise(n)
    return sample_points, samples


def fit(x, y, degree):
    model = np.poly1d(np.polyfit(x, y, degree))
    return model

def get_error(model, truth):
    return np.sum(np.square(model(truth[0]) - truth[1]))


def objective(args):
    degree = args['degree']
    print('Fitting {}', degree)
    model = fit(*DataSet.get_train(), degree=degree)
    error = get_error(model, DataSet.get_test())
    return error


def plot_error():
    X = range(100)
    Y = [get_error(fit(*DataSet.get_train(), x), DataSet.get_test()) for x in range(100)]

    plt.subplot(111)
    plt.plot(X, Y)
    plt.subplot(212)
    plt.plot(DataSet.get_train()[0], ground(DataSet.get_train()[0]))
    plt.show()


def optimize():
    space = {
        'degree': hp.quniform('degree', 20, 100, 1)
    }

    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=15)
    
    print('Best model: {}', best_model)
    print('Best error: {}', hyperopt.space_eval(space, best_model))
    print('Done')
    
    plot_error()



def run_experiment():
    optimize()
    






    

