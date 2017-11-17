# Load pandas
import pandas as pd
from matplotlib import cm

import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
    accuracy_score, auc, roc_curve, average_precision_score
#from rank_metrics import ndcg
from scipy.stats import zscore

def zscore_normalize(df, column):
    #print('mean({0}): {1}'.format(df[column].mean(), df[column].std(ddof=0)))
    df[column+'_zscore'] = (df[column] - df[column].mean()) / df[column].std(ddof=0)

def precision_recall_f1(df):
    modes = ['binary']

    precisionList = {}
    recallList = {}
    F1List = {}
    accuracyList = []
    roc_aucList = []
    averagePrecisionList = []

    for m in modes:
        precisionList[m] = []
        recallList[m] = []
        F1List[m] = []

    epsilon = 0.1
    stop = False
    k_range = range(1, df.shape[0]+1)
    for k in k_range:
        if stop:
            k_range = np.asarray( k_range)[:k-1]
            break

        if k % 500 == 0:
            print('k : ', k)

        testY = df['class']

        df['test'] = 0
        df.iloc[:k, df.columns.get_loc('test')] = 1
        predY = df['test']

        #print(testY.head(), testY.shape)
        #print(predY.head(), predY.shape)

        for m in modes:
            precision, recall, fscore, support = precision_recall_fscore_support(testY, predY, average=m, pos_label=1)

            precisionList[m].append(precision)
            recallList[m].append(recall)
            F1List[m].append(fscore)

            if 1.0 - recall <= epsilon:
                stop = True

            if k % 500 == 0:
                print ("Mode ({0}) ---> k={1:.3f}: precision={2:.3f}, recall={3:.3f}, F1={4:.3f}, accuracy={5:.3f}, sum(testY)={6}, sum(testY)={7}, sum(predY)={8}".format(m, k, precision, recall, fscore, accuracy, testY.sum(), testY[:k].sum(), predY.sum()))

        accuracy = accuracy_score(testY, predY)
        accuracyList.append(accuracy)

        fpr, tpr, thresholds = roc_curve(testY, predY, pos_label=1)
        auc_area = auc(fpr, tpr)
        roc_aucList.append(auc_area)
        if k % 500 == 0:
            print(", AUC: ", auc_area)

        averagePrecisionList.append( average_precision_score(testY, predY) )

    for m in modes:
        plt.scatter(k_range, recallList[m])
        plt.plot(k_range, recallList[m], '-x', label="Recall (" + m + ")")

        plt.scatter(k_range, precisionList[m])
        plt.plot(k_range, precisionList[m], '-o', label="Precision (" + m + ")")

        plt.scatter(k_range, F1List[m])
        plt.plot(k_range, F1List[m], '-x', label="F1 (" + m + ")")

        #plt.scatter(k_range, accuracyList)
        #plt.plot(k_range, accuracyList, '-.', label="Accuracy")

        #plt.scatter(k_range, roc_aucList)
        #plt.plot(k_range, roc_aucList, '--', label="AUC")

        plt.title("Metrics score@k")
        plt.xlabel("k")
        plt.ylabel("Score ([0,1])")

        plt.legend()

        plt.show() #savefig('c:/temp/{0}.jpg'.format(m))

        x, y, z = precisionList[m], recallList[m], F1List[m]

        fig = p.figure()
        ax = p3.Axes3D(fig)
        # plot3D requires a 1D array for x, y, and z
        # ravel() converts the 100x100 array into a 1x10000 array
        ax.plot3D(np.ravel(x), np.ravel(y), np.ravel(z))
        ax.set_xlabel('Precision')
        ax.set_ylabel('Recall')
        ax.set_zlabel('F1')
        fig.add_axes(ax)
        p.show()

    plt.scatter(k_range, accuracyList)
    plt.plot(k_range, accuracyList, '-.', label="Accuracy")

    plt.scatter(k_range, roc_aucList)
    plt.plot(k_range, roc_aucList, '--', label="AUC")

    plt.scatter(k_range, averagePrecisionList)
    plt.plot(k_range, averagePrecisionList, '--', label="AveragePrecision")

    plt.title("score@k (Accuracy & AUC)")
    plt.xlabel("k")
    plt.ylabel("Score ([0,1])")

    plt.legend()

    plt.show()


def sort_data(df, alpha=0.5):
    # normalization
    zscore_normalize(df, 'entropy')
    zscore_normalize(df, 'users')

    df["rank_score"] = df["entropy_zscore"] * alpha + df["users_zscore"] * (1 - alpha)
    #print('normalized: ')
    #print(df[["rank_score", "entropy", "entropy_zscore", "users", "users_zscore"]].head(10))
    return df.sort_values(['rank_score'], ascending=[False])


#ndcg_values = [ndcg( x for x in entropies]

def average_precision(df_sorted):
    #print(df_sorted['class'])
    k_range = range(1, df_sorted.shape[0]+1)

    precisionList = []
    recallList = []

    testY = df_sorted['class']

    for k in k_range:
        if testY[k-1] == 0:
            continue

        df_sorted['test'] = 0
        df_sorted.iloc[:k, df_sorted.columns.get_loc('test')] = 1
        predY = df_sorted['test']

        precision, recall, fscore, support = precision_recall_fscore_support(testY, predY, average='binary', pos_label=1)
        precisionList.append(precision)
        recallList.append(recall)

    #print('Precision list: ', precisionList)
    #print('Recall list: ', recallList)

    #print('Average Precision: ', np.mean( precisionList ) )
    return np.mean( precisionList )

def read_data_from_KNIME_results(filename):
    df = pd.read_csv(filename)

    d = {'yes': 1, 'no': 0}
    df['class'] = df['class'].map(d)
    df['pred_class'] = df['Prediction (class)'].map(d)
    df['users'] = df['unique_users']

    df['rank_score'] = df['P (class=yes)']

    df = df[["size", "users", "entropy", "rank_score", "class", "pred_class"]]

    return df.sort_values(['rank_score'], ascending=[False])


def handle_KNIME_results(filename):
    df_sorted = read_data_from_KNIME_results(filename)
    precision_recall_f1(df_sorted)

def process_data(df):
    # df = df.head(1000)
    print(df.head())
    print(df.count())

    bestAlpha = None
    bestAP = 0.0
    for alpha in np.arange(0.0, 1.1, 0.1):
        df_sorted = sort_data(df, alpha)
        # df_sorted.to_csv('c:/temp/deleteme/sorted.csv', sep=',')

        ap = average_precision(df_sorted)
        print('Alpha={0}, Average Precision={1}'.format(alpha, ap))
        if bestAP < ap:
            bestAlpha = alpha
            bestAP = ap

    print('best alpha between entropy and users: ', bestAlpha, ' best AP: ', bestAP)

    df_sorted = sort_data(df, bestAlpha)
    precision_recall_f1(df_sorted)


if __name__ == "__main__":
    #y_true = np.array([0, 0, 1, 1])
    #print('AP: ', average_precision_score(y_true, y_scores))

    flags = ['KNIME', 'Petrovic', 'Play']
    flag = 0

    if flags[flag] == 'KNIME':
        filename = 'C:/temp/temp_1_0.csv' # AdaBoost
        filename = 'c:/temp/RandomForest_2_Under-Sampling_0.csv' #under sampling
        handle_KNIME_results(filename)

    elif flags[flag] == 'KNIME':
        # play with data
        df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv',
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])

        process_data(df)
    elif flags[flag] == 'Play':
        # play with data

        df = pd.read_csv('C:\\temp\\deleteme\\average_precision_q2.csv', #test_precision_recall.csv', #
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])

        process_data(df)