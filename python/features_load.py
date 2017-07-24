import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import datetime


import scipy.stats as stats
import matplotlib.mlab as mlab

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def show_feature(data1, data2, title, field):
    fig, ax = plt.subplots()
    fig.suptitle(title, fontsize=16)

    a_heights, a_bins = np.histogram(data1[field], bins=100)
    b_heights, b_bins = np.histogram(data2[field], bins=a_bins)

    width = (a_bins[1] - a_bins[0])/3

    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label="Events")
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen', label="No-Events")

    ax.set_ylabel("Frequency")
    ax.set_xlabel(field)

    ax.legend(loc="upper right")

    print('--------------------------', title + " (" + field + ")")
    descriptive_stat("Event", data1[field])
    print('----')
    descriptive_stat("No-Event", data2[field])

    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    #formatter = FuncFormatter(to_percent)

    # Set the formatter
    #plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()

    x = data1[field]

    density = stats.gaussian_kde(x)
    y =  density(a_bins)
    plt.plot(a_bins, y, 'b--')

    #y = mlab.normpdf(a_bins, x.mean(), x.var())
    #plt.plot(a_bins, y, 'b--')

    plt.show()

def descriptive_stat(title, s):
    print(" >> " + title + " << ")
    print("data size : {0:8d}".format(len(s)))
    print("Mean : {0:8.6f}".format(s.mean()))
    print("Minimum : {0:8.6f}".format(s.min()))
    print("Maximum : {0:8.6f}".format(s.max()))
    print("Variance : {0:8.6f}".format(s.var()))
    print("Std. deviation : {0:8.6f}".format(s.std()))

def run_feature_analysis(data, field_titles, field_names, is_event_field='is_event', sep=',', sample_size=-1):
    start = datetime.datetime.now()

    data1 = data[data[is_event_field] == 1 ]
    data2 = data[data[is_event_field] == 0 ]

    if sample_size>0:
        sample_size = min(sample_size, len(data1))
        sample_size = min(sample_size, len(data2))

        data1 = data1.sample(sample_size) #.head(sample_size)
        data2 = data2.sample(sample_size) #.head(sample_size)

    end = datetime.datetime.now()

    for i in range(len(field_names)):
        show_feature(data1, data2, field_titles[i], field_names[i])

    print(start, end)
    print(end-start)

def visualize(data1, data2):

    for i in range(len(data1.columns)):
        for j in range(i+1, len(data1.columns)):
            if 'is_tree_event' in [ data1.columns[i] , data1.columns[j] ] \
                    or 'is_event' in [data1.columns[i], data1.columns[j]]:
                continue

            fig = plt.figure()

            title = data1.columns[i] + " vs. " + data1.columns[j]
            #plt.title(title)

            X1 = np.asarray(data1.ix[:, i])
            Y1 = np.asarray(data1.ix[:, j])
            X2 = np.asarray(data2.ix[:, i])
            Y2 = np.asarray(data2.ix[:, j])

            print(X1.shape, Y1.shape)
            print(X2.shape, Y2.shape)

            fig.suptitle(title, fontsize=20)
            plt.xlabel(data1.columns[i], fontsize=18)
            plt.ylabel(data1.columns[j], fontsize=16)
            #fig.savefig('test.jpg')

            plt.scatter(X1, Y1, c='b', marker='+', alpha=0.6)
            plt.scatter(X2, Y2, c='r', marker='o', alpha=0.6)

            plt.show()


def visualize1(data1, data2):
    # %% -*- coding: utf-8 -*-

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    X = data1.append(data2)
    target = dd['is_tree_event']

    X = X.drop('is_tree_event')
    fig = plt.figure(1, figsize=(4, 3))

    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for  label in [0, 1]:
        ax.text3D(X[y == label, 0].mean(),
                  X[y == label, 1].mean() + 1.5,
                  X[y == label, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))


    data_fit = pca.fit(dd)
    pca_res = data_fit.transform(dd)

    traces = []

    trace = Scatter(
        x=data_fit,
        y=yy,
        mode='markers',
        name=name,
        marker=Marker(
            size=12,
            line=Line(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8))
    traces.append(trace)

    data = Data(traces)
    layout = Layout(xaxis=XAxis(title='PC1', showline=False),
                    yaxis=YAxis(title='PC2', showline=False))
    fig = Figure(data=data, layout=layout)
    py.iplot(fig)




def machine_learn(data, id_field_name, y_field_name, sample_size=-1):

    data1 = data[data[y_field_name] == 1 ]
    data2 = data[data[y_field_name] == 0 ]

    if sample_size>0:
        #sample_size = min(sample_size, len(data1))
        #sample_size = min(sample_size, len(data2))
        if sample_size < len(data1) :
            data1 = data1.sample(sample_size) #.head(sample_size)

        if sample_size < len(data2):
            data2 = data2.sample(sample_size) #.head(sample_size)

    visualize(data1, data2)



    x = data1.append(data2)
    y = x[y_field_name]

    cols = [col for col in x.columns if col not in [id_field_name, y_field_name]]
    x = x[cols]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print("Train data: ", x_train.shape, "Test data: ", x_test.shape, "y train data: ", y_train.shape, "y test data: ", y_test.shape)

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis()]
    # QuadraticDiscriminantAnalysis()]
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes",
             "Linear Discriminant Analysis"]
    # "Quadratic Discriminant Analysis"]

    for name, clf in zip(names, classifiers):
        print ('*********  ', name, ' **********')

        clf.fit(x_train, y_train)
        # z_test = clf.predict(x_test)

        score = clf.score(x_test, y_test)
        print('Score using ', name, ': ', score)

        #z_predict = clf.predict(x_predict)
        #print ('Predict using ' + name + ': ')
        #for i in range(len(x_predict)):
        #    print (x_predict_dates[i] + '\t' + str(x_predict[i][i_day_week]) + '\t' + z_predict[i])

    print('finished loop!!')
    print('*' * 10)



if __name__ == "__main__":
    # folder = 'c:/temp/features_tweet.csv'
    folder = 'C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/'

    print('*' * 20)
    print('Tree features')

    data = pd.read_csv(folder + 'tree_tweet_features.csv', sep=',', index_col=0)
    run_feature_analysis(data,
                         ["Depth"],
                         ['depth'],
                         is_event_field='is_tweet_event') #, sample_size=20000000)


    data = pd.read_csv(folder + 'tree_features.csv', sep=',', index_col=0)
    run_feature_analysis(data,
                         ["Tree Size", "retweet count", 'likes average', 'Retweet Average'],
                         ['tree_size', 'rtwt_count', 'likes_avg', 'retweets_avg'],
                         is_event_field='is_tree_event', sample_size=1000)
    print('Machine learning...')
    machine_learn(data, 'root_id', 'is_tree_event', 1000)

    print('*' * 20)
    print('Clusters features')

    data = pd.read_csv(folder + 'cluster_features.csv', sep=',', index_col=0)
    run_feature_analysis(data,
                         ["Entropy feature", "#Nodes feature", "#Users feature", "Average time feature"],
                         ['entropy', 'size', 'users', 'avg_time'],
                         is_event_field='is_leader_event', sample_size=1000)
    print('Machine learning...')
    machine_learn(data, 'lead_id', 'is_leader_event', 1000)
