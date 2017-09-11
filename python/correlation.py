import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def test1(X, y):
    print('test 1')
    #df_dummies = pd.get_dummies(y)

    df_new = pd.concat([X, y], axis=1)
    x = df_new.values

    print(df_new.columns)
    correlation_matrix = np.corrcoef(x.T)
    print('correlation matrix:', correlation_matrix)


    #print('test 10')
    #df_dummies = pd.get_dummies(df['class'])
    #del df_dummies[df_dummies.columns[-1]]
    #df_new = pd.concat([df, df_dummies], axis=1)
    #del df_new['class']
    #
    #x = df_new.values
    #
    #correlation_matrix = np.corrcoef(x.T)
    #print('correlation matrix:', correlation_matrix)

#%%


def test2(X, y):
    print('test 2')

def test3(X, y):
    # Build a classification task using 3 informative features
    #X, y = make_classification(n_samples=1000,
    #                           n_features=10,
    #                           n_informative=3,
    #                           n_redundant=0,
    #                           n_repeated=0,
    #                           n_classes=2,
    #                           random_state=0,
    #                           shuffle=False)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

def test4(X, y):
    names = X.columns

    rf = RandomForestRegressor()
    rf.fit(X, y)
    print ("Features sorted by their score:")
    print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse = True))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def gini(p):
   return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

def entropy(p):
   return -1 * p * np.log2(p) - (1 - p) * np.log2(1.0 - p)

def classification_error(p):
   return 1 - np.max([p, 1 - p])

def informationGain(X, y):
    #X = np.arange(0.0, 1.0, 0.01)
    x = X['entropy']

    ent = [entropy(p) if p != 0 else None for p in x]
    scaled_ent = [e*0.5 if e else None for e in ent]
    c_err = [classification_error(i) for i in x]

    fig = plt.figure()
    ax = plt.subplot(111)

    for j, lab, ls, c, in zip(
          [ent, scaled_ent, gini(X), c_err],
          ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
          ['-', '-', '--', '-.'],
          ['lightgray', 'red', 'green', 'blue']):
        line = ax.plot(x, j, label=lab, linestyle=ls, lw=1, color=c)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85),
             ncol=1, fancybox=True, shadow=False)

    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

    plt.ylim([0, 1.1])
    plt.xlabel('p(j=1)')
    plt.ylabel('Impurity Index')
    plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def cdf(data):
    # create some randomly ddistributed data:
    #data = df['entropy'] #np.random.randn(10000)

    # sort the data:
    data_sorted = np.sort(data)

    # calculate the proportional values of samples
    p = 1. * np.arange(len(data)) / (len(data) - 1)

    # plot the sorted data:
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(p, data_sorted)
    ax1.set_xlabel('$p$')
    ax1.set_ylabel('$x$')

    ax2 = fig.add_subplot(122)
    ax2.plot(data_sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')

    plt.show()

def bin(df, column, bincount):
    m1, m2 = min(df[column]), max(df[column])
    step = 1.0 * (m2 - m1) / bincount

    bins = np.arange(m1 - step, m2 + step, step)
    return bins[:-1] + step / 2

from sklearn.naive_bayes import GaussianNB
def bayes2(X, y):
    # Create a Gaussian Classifier
    model = GaussianNB()

    # Train the model using the training sets
    model.fit(X, y)

    # Predict Output
    bins = bin(X, 'entropy', 10)
    predicted = model.predict(bins)

    print(predicted)

import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def Univariate(X, Y):
    # load data
    #url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    #names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    #dataframe = pandas.read_csv(url, names=names)
    #array = dataframe.values
    #X = array[:, 0:8]
    #Y = array[:, 8]
    # feature extraction
    print('Univariate:')
    test = SelectKBest(score_func=chi2, k=3)
    fit = test.fit(X, Y)
    # summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_.astype('|S10'))
    features = fit.transform(X)
    # summarize selected features
    print(features[0:5, :])


from sklearn.decomposition import PCA

def PCA_based(X, y):
    # feature extraction
    print('PCA:')
    pca = PCA(n_components=3)
    fit = pca.fit(X)
    # summarize components
    print("Explained Variance: ", fit.explained_variance_ratio_)
    print(fit.components_)

def bayes(df, column, bincount):
    group_names = bin(df, column, bincount)

    df[column+'_bin'] = pd.cut(df[column], bins, labels=group_names)

    for b in group_names:
        df.loc[df[column+'_bin'] == b]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from pandas import read_csv

def LR_based(X, y):
    #feature extraction
    print('LogisticRegression')
    model = LogisticRegression()
    rfe = RFE(model, 3)
    fit = rfe.fit(X, y)
    print("Num Features: ", fit.n_features_)
    print("Selected Features: ", fit.support_)
    print("Feature Ranking: %s", fit.ranking_)

def feature_importance(X, y):
    # feature extraction
    print('feature_importance')
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)

if __name__ == "__main__":
    df = pd.read_csv('c:/temp/data_minimal_features.csv')
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    X = df.ix[:, df.columns != 'class']

    print(df.columns)

    d = {'yes': 1, 'no': 0}
    y = df['class'].map(d)
    # y = y == 'yes'
    # y[ y['class'] == 'yes'] = 1
    # y[ y['class'].strip() == 'no'] = 0

    #X = X.head(7000)
    #y = y.head(7000)

    #test1(X, y)
    #test2(X, y)
    #test3(X, y)

    #test4(X, y)

    #informationGain(X, y)

    yes = df.loc[df["class"] == 'yes']
    no = df.loc[df["class"] == 'no']

    #print(yes.head(10))
    #print(df.head(10))

    tmp1 = yes['entropy']
    tmp1 = no['entropy']

    #cdf(tmp1)
    #cdf(tmp2)

    #bayes(df, 'entropy', 10)

    #print(df.head(100))
    #bayes2(X, y)

    #Univariate(X, y)
    #PCA_based(X, y)
    LR_based(X, y)

    #feature_importance(X, y)
    #test3(X, y)
    #test4(X, y)

    test1(X, y)