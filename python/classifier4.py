# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 18:22:07 2016

@author: SAMER AAMAR
"""
import numpy as np
import matplotlib.pyplot as plt
import time


# Parameters
iris_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# iris_url = 'iris_dataset/iris.data'
breast_cancer_url = 'c:/temp/bc/wpbc.data'  # https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
# breast_cancer_url = 'breast_cancer/wpbc.data'
logger_level = 1  # 1 = info, 0 = debug
# plot_vis = True      # plot graphs or not
test_ratio = 0.2


# %%

# -----------------------------------------
# helper functions
def info(message):
    if logger_level <= 1:
        print(message)


def debug(message):
    if logger_level <= 0:
        print(message)


# %%

# from pandas.tools.plotting import scatter_matrix #, parallel_coordinates
import pandas as pd


def getdata():
    df = pd.read_csv('C:\\temp\\Sep14\\cluster_by_lead_id_bonus_users.csv') #'C:/temp/data-cluster1.csv') #c:/temp/data_minimal_features.csv')
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    #feature_col = ['entropy', 'unique_users', 'size']
    X = df [df.columns != 'class']

    d = {'yes': 1, 'no': 0}
    Y = df['class'].map(d)

    X = X.as_matrix()
    Y = Y.as_matrix().astype(int)

    return X, Y

def prepare_breast_cancer_ds():
    names = ['ID', 'outcome', 'time']
    tmp = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points',
           'symmetry', 'fractal_dimension']

    for j in range(1, 4):
        names = names + [tmp[i] + '_' + str(j) for i in range(10)]

    names = names + ['Tumor size', 'Lymph node status']

    dataframe = pd.read_csv(breast_cancer_url, names=names)
    dataframe = dataframe[dataframe['Lymph node status'] != '?']

    Y = dataframe[['outcome']].copy()
    Y.loc[Y.outcome != 'N', 'outcome'] = 1.0
    Y.loc[Y.outcome == 'N', 'outcome'] = 0.0

    columns = names
    columns.remove('ID')
    columns.remove('outcome')
    columns.remove('time')

    total = dataframe.shape[0]
    X = dataframe[columns].copy()

    info("Breast Cancer Dataset: selected {0} entries out of {1} from the dataset".format(X.shape, total))

    X = X.as_matrix()
    Y = Y.as_matrix().astype(int)

    return X, Y


# %%
import itertools


def prepare_iris_DS():
    info("Loading iris DS...")
    iris = pd.read_csv(iris_url, names=["sepal length", "sepal width", "petal length", "petal width", "Species"])
    df = pd.DataFrame(iris, columns=["sepal length", "sepal width", "petal length", "petal width", "Species"])

    df.head()
    iris.head()

    # global plot_vis
    # if plot_vis:
    # plt.figure()

    # andrews_curves(iris, 'Species')
    # scatter_matrix(iris, alpha=0.2, figsize=(6, 6), diagonal='kde', marker="x")
    # plt.savefig(graphs_folder + r"/plt_iris_ds.jpg")


    info("Iris DS is Loaded")

    columns, labels = ["sepal length", "sepal width", "petal length", "petal width"], ['Iris-virginica',
                                                                                       'Iris-versicolor']

    # the following are separable:
    # ["Iris-setosa", "Iris-virginica"]

    total = df.shape[0]
    df = df[df.Species.isin(labels)]
    X = df[columns].copy()

    info("IRIS Dataset: selected {0} samples out of {1} from the dataset based on labels {2}".format(X.shape, total,
                                                                                                     str(labels)))

    Y = df[["Species"]].copy()
    Y.loc[Y.Species != labels[0], 'Species'] = 0.0
    Y.loc[Y.Species == labels[0], 'Species'] = 1.0

    X = X.as_matrix()
    Y = Y.as_matrix().astype(float)

    return X, Y


# The following method is taken as is from internet
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        info("Normalized confusion matrix")
    else:
        info('Confusion matrix, without normalization')

    info(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'  # if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%
# from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D

# from http://matplotlib.org/examples/mplot3d/
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

# from mpl_toolkits.mplot3d import Axes3D
# from http://matplotlib.org/examples/mplot3d/
import matplotlib.pyplot as plt


def plot_trisuf3d(mX, mY, mZ, title, xlabel, ylabel, zlabel):
    x = np.asarray(mX)
    y = np.asarray(mY)
    z = np.asarray(mZ)

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    ax.set_title(title)
    xmin = np.min(mX)
    ymin = np.min(mY)
    xmax = np.max(mX)
    ymax = np.max(mY)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(0, min(2.0 * max(mZ), 1.0))

    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    plt.close()


# %%
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix


def my_train_test_split(X, Y, test_size, random_state=None):
    # reuse existing function to split the dataset. random_state means it is real random
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    # msk = np.random.rand(X.shape[0]) < (1.0 - test_size)
    # X_train = X[msk]
    # X_test = X[~msk]
    # y_train = Y[msk]
    # y_test = Y[~msk]

    return X_train, X_test, y_train.ravel(), y_test.ravel()


# %%
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %% Train SVM model to find the best gamma and C
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def run_svm(X, Y, test_size, gamma, C):
    X_train, X_test, y_train, y_test = my_train_test_split(X, Y, test_size=test_size)
    # svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C)

    rbf_svc.fit(X_train, y_train)

    # nn = neural_network.BernoulliRBM(learning_rate=0.1, n_iter=10)
    # nn.fit(X_train, y_train)
    score = rbf_svc.score(X_test, y_test)

    y_pred = rbf_svc.predict(X_test)
    # debug (rbf_svc.predict(v))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)

    debug(cnf_matrix)

    return cnf_matrix, score


# %%


def find_best_svm_params_helper(X, Y, tries, test_size, C_range=(0.1, 10, 0.5), gamma_range=(0.1, 10, 0.5)):
    mX = []
    mY = []
    mZ = []

    best_score = 0
    best_C = 0
    best_gamma = 0

    C_start, C_stop, C_step = C_range
    gamma_start, gamma_stop, gamma_step = gamma_range

    c = 0
    C_range = np.arange(C_start, C_stop, C_step)
    gamma_range = np.arange(gamma_start, gamma_stop, gamma_step)

    total = len(C_range) * len(gamma_range)
    debug('Total tries: {0}'.format(total))
    debug('#\t%\tC\tgamma\tscore')
    for C in C_range:
        for gamma in gamma_range:

            avg = 0.0
            for runs in range(tries):
                tmp, score = run_svm(X, Y, test_size, gamma, C)

                if avg == 0.0:
                    cnf_matrix_tmp = tmp
                else:
                    cnf_matrix_tmp = tmp + cnf_matrix_tmp

                avg = avg + score

            avg /= float(tries)
            if best_score < avg:
                best_score = avg
                best_C = C
                best_gamma = gamma
                # calculate avergare of confusion matrix
                cnf_matrix = cnf_matrix_tmp / float(tries)

            if (float(c) / total * 100) - int(float(c) / total * 100) == 0:
                debug('{3}\t{4:0.0f}%\t{0}\t{1}\t{2}'.format(C, gamma, avg, c, 100 * c / total))
            c += 1
            mX.append(C)
            mY.append(gamma)
            mZ.append(avg)

    plot_trisuf3d(mX, mY, mZ, 'SVM', 'Gamma', 'C', 'Score')

    return best_C, best_gamma, best_score, cnf_matrix


# %%

def find_best_svm_params(X, Y, epsilon=0.001, tries=10, test_size=0.2):
    starttime = time.time()
    best_score = 0
    C_range = (1, 1000, 100)
    gamma_range = (1, 1000, 100)
    for i in range(10):
        debug(str(C_range) + str(gamma_range))
        new_C, new_gamma, new_score, new_cnf_matrix = find_best_svm_params_helper(X, Y, tries, test_size,
                                                                                  C_range=C_range,
                                                                                  gamma_range=gamma_range)

        d = 0
        if new_score > best_score:
            d = new_score - best_score
            best_C = new_C
            best_gamma = new_gamma
            best_score = new_score
            cnf_matrix = new_cnf_matrix

        info('Trying C range {3} and gamma range {4}. best C: {0}, best gamma: {1}, best score: {2}'.format(best_C,
                                                                                                            best_gamma,
                                                                                                            best_score,
                                                                                                            C_range,
                                                                                                            gamma_range))
        if d < epsilon:
            # we are close enough, let's stop
            debug('Break')
            break

        C_range = (max(0.1, best_C - C_range[2]), best_C + C_range[2], C_range[2] / 10.0)
        gamma_range = (max(0.1, best_gamma - gamma_range[2]), best_gamma + gamma_range[2], gamma_range[2] / 10.0)

    info('best C: {0}, best gamma: {1}, best score: {2}'.format(best_C, best_gamma, best_score))

    endtime = time.time()

    delta = endtime - starttime
    print("Elapsed time: {0:0.0f}:{1:0.0f} (total in seconds: {2})".format(delta / 60, delta % 60, delta))

    return best_C, best_gamma, best_score, cnf_matrix


# %%

def start_svm(X, Y, class_names, test_size, tries=10):
    info('Search for best parameters for SVM.')
    info('----------------------------------.')
    best_C, best_gamma, best_score, cnf_matrix = find_best_svm_params(X, Y, tries=tries, test_size=test_size)

    # Plot non-normalized confusion matrix
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix (SVM), without normalization')
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix (SVM), without normalization')
    info('best C: {0}, best gamma: {1}, best score: {2}'.format(best_C, best_gamma, best_score))

    model = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
    return model


# %%
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %% Train Neural Network model to find the best number of layers with best learning rate (as a constant)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# %%

from sklearn.neural_network import MLPClassifier


def start_neural_network(X, Y, test_size):
    info('Search for best parameters for neural network.')
    info('---------------------------------------------.')
    layer = []
    rates = []
    performance = []
    score = []

    times = 10

    best_score = 0
    best_rate = None
    best_layers = None
    rate_range = np.arange(1.0 / 50, 1, 1.0 / 50)
    for l in range(2, 50):
        for rate in rate_range:
            p = 0.0
            s = 0.0
            for i in range(times):
                X_train, X_test, y_train, y_test = my_train_test_split(X, Y, test_size=test_size)  # , random_state=i)

                starttime = time.time()
                mlp = MLPClassifier(hidden_layer_sizes=(l,),
                                    # max_iter=10,
                                    solver='sgd',
                                    learning_rate='constant',
                                    learning_rate_init=rate)
                # mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                #                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                #                    learning_rate_init=.1)

                mlp.fit(X_train, y_train)
                endtime = time.time()
                p += endtime - starttime
                s += mlp.score(X_test, y_test)

            layer.append(l)
            rates.append(rate)
            performance.append(p / times)

            s = s / times
            score.append(s)
            if (s > best_score):
                best_score = s
                best_rate = rate
                best_layers = l

                # print("Training set score: %f" % mlp.score(X_train, y_train))
                # print("Test set score: %f" % s)

    plot_trisuf3d(layer, rates, score, 'Neural Network: Scores', 'Layers', 'Rate', 'Score')
    plot_trisuf3d(layer, rates, performance, 'Neural Network: Performance', 'Layers', 'Rate', 'Seconds')
    info('Print results for neural network search')
    info('Best score is {0}. Recommended rate: {1} and recommended layer: {2}'.format(best_score, best_rate,
                                                                                      best_layers))

    mlp = MLPClassifier(hidden_layer_sizes=(best_layers,),
                        # max_iter=10,
                        solver='sgd',
                        learning_rate='constant',
                        learning_rate_init=best_rate)

    return mlp


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# %% compare between SVM and
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# evaluate each model in turn
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import accuracy_score
# from sklearn import cross_validation

def compare(X, Y, models, classnames, title, test_size=0.2):
    # prepare configuration for cross validation test harness

    # num_folds = 2
    # num_instances = len(X)
    # seed = 42

    results = []
    names = []
    for name, model in models:
        # print ('model ....', name, model)

        # seed += 1

        # tmpscore = cross_val_score(model, X, Y.ravel(), scoring='accuracy')
        # print('score is: ', tmpscore)
        cv_results = []
        iterations = 10
        fit_time = test_time = 0.0
        for i in range(iterations):
            starttime = time.time()
            X_train, X_test, y_train, y_test = my_train_test_split(X, Y, test_size)  # , random_state=(i+1)*seed)
            model.fit(X_train, y_train)
            endtime = time.time()
            fit_time += endtime - starttime

            starttime = time.time()
            y_pred = model.predict(X_test)
            # cv_results.append ( accuracy_score(y_pred , y_test) )
            endtime = time.time()
            test_time += endtime - starttime

            cv_results.append(model.score(X_test, y_test))

            tmp = confusion_matrix(y_test, y_pred)
            if i == 0:
                cnf_matrix = tmp
            else:
                cnf_matrix = cnf_matrix + tmp
                # Compute confusion matrix
                # cv_results.append( model.score(X_test, y_test) )

        info('For model {0}: fit time is {1} (s), test time is {2} (s) for {3} iterations'.format(name, fit_time,
                                                                                                  test_time,
                                                                                                  iterations))  # average times
        cnf_matrix = cnf_matrix / float(iterations)

        plot_confusion_matrix(cnf_matrix, classnames, title=title + ': ' + name)

        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, np.mean(cv_results), np.std(cv_results))
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


# %%

datasets = []

iris_ds = False
if iris_ds:
    info('IRIS Dataset')
    X, Y = prepare_iris_DS()

    classnames = ['Iris-virginica', 'Iris-versicolor']
    datasets.append(("IRIS Dataset", X, Y, classnames))

breast_cancer = False
if breast_cancer:
    X, Y = prepare_breast_cancer_ds()
    classnames = ['DiseaseFree', 'Infected']
    datasets.append(('Breast Cancer Dataset', X, Y, classnames))

ned = False
if ned:
    X, Y = getdata()
    classnames = ['Event', 'Not-Event']
    datasets.append(('Event Detection', X, Y, classnames))

# %%


models = {}
for title, X, Y, classnames in datasets:
    svm_time = mlp_time = 0
    print('==============TRAIN on DATA SET ({0})======================='.format(title))
    starttime = time.time()
    svm_model = start_svm(X, Y, classnames, test_size=test_ratio)
    endtime = time.time()
    svm_time += endtime - starttime
    models[title] = []
    models[title].append(["SVM", svm_model])

    starttime = time.time()
    mlp = start_neural_network(X, Y, test_size=test_ratio)
    endtime = time.time()
    mlp_time += endtime - starttime
    models[title].append(["NN", mlp])
    print('Time for SVM parameter search: {0} seconds'.format(svm_time))
    print('Time for NN  parameter search: {0} seconds'.format(mlp_time))
# %%
# compare between the models
for title, X, Y, classnames in datasets:
    info('==============DATA SET ({0})======================='.format(title))
    times = compare(X, Y, models[title], classnames, title, test_size=test_ratio)
