# Load pandas
import pandas as pd
from matplotlib import cm
from sklearn.metrics import confusion_matrix
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

def calculateScoreAt_k(df, k_range, score_types=['f1', 'precision']):
    trueY = df['class']

    scores = {}
    for m in score_types:
        scores[m] = []

    for k in k_range:
        if k % 500 == 0:
            print('k : ', k)

        df['test'] = 0
        df.iloc[:k, df.columns.get_loc('test')] = 1
        predY = df['test']

        precision, recall, fscore, support = precision_recall_fscore_support(trueY, predY, average='binary', pos_label=1)

        if 'f1' in score_types:
            scores['f1'].append(fscore)

        if 'precision' in score_types:
            scores['precision'].append(precision)

    return scores

def calculateScoreAt_k_over_alpha(df, kList, alpha_values, m = 'f1'):
    bestK = -1
    best_Score = None
    scores = {}

    for alpha in alpha_values:
        df_sorted = sort_data(df, alpha)

        temp = calculateScoreAt_k(df_sorted, kList, [m])
        scores[alpha] = temp[m]

        score = max(scores[alpha])
        if best_Score is None or best_Score > score:
            k = scores[alpha].index(score)  # np.argmin(scores)
            bestK, best_Score, best_alpha = kList[k], score, alpha

        print('  (alpha{0:.2f}) we get best score (score={2:.7f}) at (K={1})'.format(alpha, kList[k], score))

    return scores


def detectionErrorTradeOff(df, m='DEF', cMiss=1.0, cFA=0.1, plt_ax=None, maxK=None):
    ax = plt_ax
    detList = {}
    norm1DetList = {}
    norm2DetList = {}

    detList[m] = []
    norm1DetList[m] = []
    norm2DetList[m] = []

    stop = False
    if maxK is None:
        maxK = df.shape[0]+1
    k_range = np.arange(1, maxK, 10)
    ## threshold  is k

    testY = df['class']

    trueValues  = testY[testY == 1].count()
    falseValues = testY[testY == 0].count()

    pTarget = trueValues / testY.count() # 0.01
    pNonTarget = 1.0-pTarget #falseValues / testY.count() # 1-pTarget

    print('run on k<={0}, P(target)={1:0.7f}'.format(maxK, pTarget))
    assert (pNonTarget == 1-pTarget)

    normalizer1 = min(cMiss * pTarget, cFA * (1 - pTarget))
    #normalizer2 = max(cMiss * pTarget, cFA * (1 - pTarget))

    kList = []
    bestK, best_cDET = 0, [100000000000]

    print('         starting iteration: loop over ', len(k_range), ' different k values.')
    for k in k_range:
        if k % 10 == 0:
            print('threshold = ', k)

        if k > maxK :
            stop = True

        if stop:
            #k_range = np.asarray( k_range)[:k-1] #align sizes
            break

        kList.append(k)

        df['test'] = 0
        df.iloc[:k, df.columns.get_loc('test')] = 1
        predY = df['test']

        #print(testY.head(), testY.shape)
        #print(predY.head(), predY.shape)

        CM = confusion_matrix(testY, predY, labels=[0, 1])

        #tn = CM[0][0]
        #fn = CM{1, 0}
        #tp = CM{1, 1}
        #fp = CM{0, 1}

        [[tn, fp], [fn, tp]] = CM

        pMiss = fn / (fn + tp) # miss rate (1-recall)
        pFA = fp / (fp + tn) # false alarm rate

        cDET = cMiss * pMiss * pTarget + cFA * pFA * pNonTarget
        cDETnorm1 = cDET / normalizer1
        #cDETnorm2 = cDET / normalizer2
        norm1DetList[m].append(cDETnorm1)
        #norm2DetList[m].append(cDETnorm2)

        detList[m].append(cDET)

        if best_cDET[0] > cDET:
            best_cDET = cDET, cDETnorm1
            best_K = k

    if ax is None:
        fig, ax = plt.subplots(1, 2)

    k_range = kList
    ax[0].scatter(k_range, detList[m])
    ax[0].plot(k_range, detList[m], '-x', label="DET (" + m + ")")
    ax[0].legend(loc="upper right")

    ax[0].set_title("DET@k")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Score ([0,1])")

    ax[1].scatter(k_range, norm1DetList[m])
    ax[1].plot(k_range, norm1DetList[m], '-o', label="normDET (" + m + ")")
    ax[1].legend(loc="upper right")

    ax[1].set_title("normalized DET@k")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Score ([0,1])")

    plt.suptitle(title)
    plt.legend()

    if plt_ax is None:
        plt.show()

    return best_K, best_cDET


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score



def precision_recall_f1(df, title):
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
    k_range = np.linspace(1, df.shape[0], 20, dtype=int)
    #k_range = range(1, df.shape[0]+1)
    for k in k_range:
        if stop:
            new_size = len(recallList[m])
            k_range = np.asarray( k_range)[:new_size]
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

            #**********************************************************
            average_precision = average_precision_score(testY, predY)

            print('Average precision-recall score: {0:0.2f}'.format(average_precision))

            precision, recall, _ = precision_recall_curve(testY, predY)

            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision ))

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

        plt.title("Metrics score@k (" + title + ")")
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

        fig = p.figure()
        plt.scatter(k_range, accuracyList)
        plt.plot(x, y, '-.', label="Precision vs. Recall")
        plt.show()

    plt.scatter(k_range, accuracyList)
    plt.plot(k_range, accuracyList, '-.', label="Accuracy")

    plt.scatter(k_range, roc_aucList)
    plt.plot(k_range, roc_aucList, '--', label="AUC")

    plt.scatter(k_range, averagePrecisionList)
    plt.plot(k_range, averagePrecisionList, '--', label="AveragePrecision")

    plt.title("score@k (Accuracy & AUC): " + title)
    plt.xlabel("k")
    plt.ylabel("Score ([0,1])")

    print('Best AP: ' , np.max(averagePrecisionList))


    plt.legend()

    plt.show()

def sort_data(df, alpha=0.5):
    #print(np.argmin(df.applymap(np.isreal).all(1)))
    #temp = np.where(np.any(np.isnan(df.convert_objects(convert_numeric=True)), axis=1))

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

def read_csv(filename):
    df = pd.read_csv(filename)

    d = {'yes': 1, 'no': 0}
    df['class'] = df['class'].map(d)
    df['pred_class'] = df['Prediction (class)'].map(d)
    df['users'] = df['unique_users']

    return df


def read_data_from_KNIME_results(filename):
    df = read_csv(filename)
    df['rank_score'] = df['P (class=yes)']

    df = df[["size", "users", "entropy", "rank_score", "class", "pred_class"]]

    return df.sort_values(['rank_score'], ascending=[False])


def handle_KNIME_results(filename):
    df_sorted = read_data_from_KNIME_results(filename)
    precision_recall_f1(df_sorted, filename)

def find_best_alpha_for_AP(df, title):
    # df = df.head(1000)
    #print(df.head())
    print(len(df) , ' rows')

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

    return bestAlpha


if __name__ == "__main__":
    #y_true = np.array([0, 0, 1, 1])
    #print('AP: ', average_precision_score(y_true, y_scores))

    flags = ['KNIME', 'Petrovic-scores', 'Petrovic-DET', 'Petrovic-F1', 'Play', 'F1,Precision,Recall', 'Petrovic-F1,MAP,Precision,Recall scores']

    flag = 6

    path = 'C:/temp/deleteme/classification.tweet_based_old_timestamp_calc/'
    files = ['adaboost.1.SMOTE.csv', 'bagging_R1.1.SMOTE.csv', 'bagging_dtree.1.SMOTE.csv', 'randomforest.1.SMOTE.csv',
             'adaboost.2.Under-Sampling.csv', 'bagging_R1.2.Under-Sampling.csv', 'bagging_dtree.2.Under-Sampling.csv', 'randomforest.2.Under-Sampling.csv']

    print('Working on ', flags[flag])
    if flags[flag] == 'KNIME':
        filename = 'C:/temp/temp_1_0.csv' # AdaBoost
        filename = 'c:/temp/RandomForest_2_Under-Sampling_0.csv' #under sampling
        handle_KNIME_results(filename)

    elif flags[flag] == 'Petrovic-scores':
        # play with data
        df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv', #tweets_topics3.samer.csv' , #,
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])

        process_data(df)

    elif flags[flag] == 'Petrovic-F1,MAP,Precision,Recall scores':
        filename = files[0] #take one of the files and use it as Petrovic does: 1.entropy+user 2.timestamp
        title = 'Rank by growth'
        print('.3.....' * 10)
        fig, ax = plt.subplots(1, 1)
        df = read_csv(path + filename)
        bestK, best_cDET = 0, [100000000000]
        for sortBy in ["p_20k", "p_40k", "p_40k", "p_80k"]:
            df["rank_score"] = df[sortBy]
            df_sorted = df.sort_values(['rank_score'], ascending=[False])
            precision_recall_f1(df_sorted, '{0} (rank by {1})'.format(title, sortBy))
            #K, cDET = detectionErrorTradeOff(df_sorted, 'sort by={0}'.format(sortBy), cMiss=cMiss, cFA=cFA, plt_ax=ax,
            #                                 maxK=maxK)
            #if best_cDET[0] > cDET[0]:
            #    bestK, best_cDET, best_sortBy = K, cDET, sortBy
            #print('For sortBy={0}, we get: K={1}, cDET={2:.7f} (norm={3:.7f}'.format(sortBy, K, cDET[0], cDET[1]))

        #print(' ************* best K={0}, best cDET={1:.7f} (norm={4:.7f}, best sortby={2} [{3}]'.format(bestK,
        #                                                                                                 best_cDET[0],
        #                                                                                                 best_sortBy,
        #                                                                                                 title,
        #                                                                                                 best_cDET[1]))
        plt.show()

        for filename in files:
            print('Calc scores: ', filename)

            # now try the P(class=yes)
            print('.1....' * 10)
            print('try p(class=yes)... ', filename)
            df_sorted = read_data_from_KNIME_results(path + filename)
            precision_recall_f1(df_sorted, 'sort by P(class=yes): ' + filename)

        print('.2.....' * 10)
        filename = files[0] #take one of the files and use it as Petrovic does: 1.entropy+user 2.timestamp
        title = 'Rank by entropy+user '
        print(title)
        df = read_csv(path + filename)

        bestAlpha = find_best_alpha_for_AP(df, 'find best alpha for AP')
        df_sorted = sort_data(df, bestAlpha)

        precision_recall_f1(df_sorted, '{0} (alpha={1}'.format(title, bestAlpha))


    elif flags[flag] == 'Petrovic-DET':
        cMiss = 1.0
        cFA = 0.1
        maxK = 500
        #alpha_values = np.arange(0.001, 1, 0.1)
        alpha_values = np.linspace(0.001, 1.0, 10)

        # #########################################
        # print('(NOT REALLY) : take top growing fast', end='')
        # df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics3.samer.csv',
        #                  header=None,
        #                  names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text", "p_70k", "p_50k", "p_90k"])
        # print('.... data loaded successfully: {0} rows'.format(df.shape[0]))
        # fig, ax = plt.subplots(1, 2)
        # bestK, best_cDET = 0, 100000000000
        # for alpha in alpha_values:
        #     df_sorted = sort_data(df, alpha)
        #
        #     cMiss = 1.0
        #     cFA = 0.1
        #     K, cDET = detectionErrorTradeOff(df_sorted, 'alpha={0:.2f}'.format(alpha), cMiss=cMiss, cFA=cFA, plt_ax=ax)
        #     if best_cDET > cDET:
        #         bestK, best_cDET, best_alpha = K, cDET, alpha
        #     print('For alpha={0:.2f}, we get: K={1}\tcDET={2:.7f}'.format(alpha, K, cDET))
        #
        # print('best K={0}, best cDET={1:.7f}, best alpha={2:.4f}'.format(bestK, best_cDET, best_alpha))
        # plt.show()

        # #########################################
        # fig, ax = plt.subplots(1, 2)
        # title = 'Without ranking by timestamp: tweets_topics2.csv'
        # print('Search cost function for ' + title, end='')
        # df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv' , #tweets_topics3.samer.csv',
        #                  header=None,
        #                  names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])
        # print('.... data loaded successfully: {0} rows'.format(df.shape[0]))
        # bestK, best_cDET = 0, [100000000000]
        #
        # for alpha in alpha_values:
        #     df_sorted = sort_data(df, alpha)
        #     K, cDET = detectionErrorTradeOff(df_sorted, 'tweets_topics2.csv: alpha={0:.2f}'.format(alpha), cMiss=cMiss, cFA=cFA, plt_ax=ax, maxK=maxK)
        #     if best_cDET[0] > cDET[0]:
        #         bestK, best_cDET, best_alpha = K, cDET, alpha
        #     print('For alpha={0:.2f}, we get: K={1}\tcDET={2:.7f}\tcDETnorm={3:.7f}'.format(alpha, K, cDET[0], cDET[1]))
        #
        # print(' ************* K={0}, best cDET={1:.7f} (norm={2:.7f}), best alpha={3:.4f} [{4}]'.format(bestK, best_cDET[0], best_cDET[1], best_alpha, title))
        # plt.show()


        # #########################################
        fig, ax = plt.subplots(1, 2)
        title = 'User+Entropy (best alpha): adaboost2.ranking.csv'
        print('Search cost function for ' + title, end='')
        df = pd.read_csv('c:/temp/deleteme/classification/randomforest.1.SMOTE.csv' ,
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])
        print('.... data loaded successfully: {0} rows'.format(df.shape[0]))
        bestK, best_cDET = 0, [100000000000]

        maxK = 100

        for alpha in alpha_values:
            print('alpha = ', alpha)
            df_sorted = sort_data(df, alpha)
            K, cDET = detectionErrorTradeOff(df_sorted, 'alpha={0:.2f}'.format(alpha), cMiss=cMiss, cFA=cFA, plt_ax=ax, maxK=maxK)
            if best_cDET[0] > cDET[0]:
                bestK, best_cDET, best_alpha = K, cDET, alpha
            print('For alpha={0:.2f}, we get: K={1}\tcDET={2:.7f}\tcDETnorm={3:.7f}'.format(alpha, K, cDET[0], cDET[1]))

        print(' ************* K={0}, best cDET={1:.7f} (norm={2:.7f}), best alpha={3:.4f} [{4}]'.format(bestK, best_cDET[0], best_cDET[1], best_alpha, title))
        plt.show()

        # #########################################
        fig, ax = plt.subplots(1, 2)
        for filename in files:
            title = '{0} - Rank by P(class=yes)'.format(filename)
            filename = path + filename

            print(title, end='')
            df = read_data_from_KNIME_results(filename)
            print('.... data loaded successfully: {0} rows'.format(df.shape[0]))
            K, cDET = detectionErrorTradeOff(df, title, cMiss=cMiss, cFA=cFA, plt_ax=ax, maxK=maxK)

            print(' ************* best K={0}, best cDET={1:.7f} (norm {2:.7f} [{3}]'.format(K, cDET[0], cDET[1], title))

        plt.show()

        # #########################################

        for filename in files:
            title = 'Rank by fastest growing {0}'.format(filename)
            filename = path + filename

            print(title, end='')

            df = read_csv(filename)
            print('.... data loaded successfully: {0} rows'.format(df.shape[0]))

            fig, ax = plt.subplots(1, 2)
            bestK, best_cDET = 0, [100000000000]
            for sortBy in ["p_20k", "p_40k", "p_40k", "p_80k"]:
                df["rank_score"] = df[sortBy]
                df_sorted = df.sort_values(['rank_score'], ascending=[False])
                K, cDET = detectionErrorTradeOff(df_sorted, 'sort by={0}'.format(sortBy), cMiss=cMiss, cFA=cFA, plt_ax=ax,
                                                 maxK=maxK)
                if best_cDET[0] > cDET[0]:
                    bestK, best_cDET, best_sortBy = K, cDET, sortBy
                print('For sortBy={0}, we get: K={1}, cDET={2:.7f} (norm={3:.7f}'.format(sortBy, K, cDET[0], cDET[1]))

            print(' ************* best K={0}, best cDET={1:.7f} (norm={4:.7f}, best sortby={2} [{3}]'.format(bestK, best_cDET[0], best_sortBy, title, best_cDET[1]))
            plt.show()

        # #########################################
        title = 'Rank by fastest growing: tweets_topics2.fast_grow.csv'
        print('Search cost for ' + title, end='')
        df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.fast_grow.csv', #tweets_topics3.samer.csv',
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text", "p_20k", "p_40k", "p_40k", "p_80k"])
        print('.... data loaded successfully: {0} rows'.format(df.shape[0]))
        fig, ax = plt.subplots(1, 2)
        bestK, best_cDET = 0, [100000000000]
        for sortBy in ["p_20k", "p_40k", "p_40k", "p_80k"]:
            df["rank_score"] = df[sortBy]
            df_sorted = df.sort_values(['rank_score'], ascending=[False])
            K, cDET = detectionErrorTradeOff(df_sorted, 'sort by={0}'.format(sortBy), cMiss=cMiss, cFA=cFA, plt_ax=ax, maxK=maxK)
            if best_cDET[0] > cDET[0]:
                bestK, best_cDET, best_sortBy = K, cDET, sortBy
            print('For sortBy={0}, we get: K={1}, cDET={2:.7f} (norm={3:.7f}'.format(sortBy, K, cDET[0], cDET[1]))

        print(' ************* best K={0}, best cDET={1:.7f} (norm={4:.7f}, best sortby={2} [{3}]'.format(bestK,
                                                                                                         best_cDET[0],
                                                                                                         best_sortBy,
                                                                                                         title,
                                                                                                         best_cDET[1]))
        plt.show()

    elif flags[flag] == 'F1,Precision,Recall':


        for filename in files:
            title = 'Rank by entropy+user {0}'.format(filename)
            filename = path + filename

            print(title, end='')

            df = read_csv(filename)


            print(title, end='')

            read_csv(filename)
            maxK = df.shape[0] + 1
            kList = np.arange(1, maxK, 10)
            alpha_values = np.arange(0.0, 1.1, 0.1)

            scores = calculateScoreAt_k_over_alpha(df, kList, alpha_values, 'precision')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y = np.meshgrid(alpha_values, kList)

            Z = np.array([scores[x][y] for x, y in zip(np.ravel(alpha_values), range(len(kList)))])
            #Z = zs.reshape(X.shape)
            ax.plot_surface(X, Y, Z)

            ax.set_xlabel('alpha')
            ax.set_ylabel('k')
            ax.set_zlabel('score(F1)')
            #ax.legend(loc="upper right")
            #ax.set_title(s+"@k")

            #plt.legend()
            plt.show()

            bestF1 = None
            for x in alpha_values:
                for y in range(len(kList)):
                    if bestF1 is None or bestF1 < scores[x][y]:
                        bestF1 = scores[x][y]
                        bestK = kList[y]
                        best_alpha = x

            print(' ************* K={0}, best score={1:.7f}, best alpha={2:.2f}'.format(bestK, bestF1, best_alpha))
            #plt.show()






    elif flags[flag] == 'Petrovic-F1':
        print('Rank by entropy+user ', end='')

        df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv' , #tweets_topics3.samer.csv',
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])
        print('.... data loaded successfully: {0} rows'.format(df.shape[0]))

        #df = df.head(500)

        maxK = df.shape[0] + 1
        kList = np.arange(1, maxK, 10)
        alpha_values = np.arange(0.0, 1.1, 0.1)


        scores = calculateScoreAt_k_over_alpha(df, kList, alpha_values, 'precision')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(alpha_values, kList)

        Z = np.array([scores[x][y] for x, y in zip(np.ravel(alpha_values), range(len(kList)))])
        #Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('alpha')
        ax.set_ylabel('k')
        ax.set_zlabel('score(F1)')
        #ax.legend(loc="upper right")
        #ax.set_title(s+"@k")

        #plt.legend()
        plt.show()

        bestF1 = None
        for x in alpha_values:
            for y in range(len(kList)):
                if bestF1 is None or bestF1 < scores[x][y]:
                    bestF1 = scores[x][y]
                    bestK = kList[y]
                    best_alpha = x

        print(' ************* K={0}, best score={1:.7f}, best alpha={2:.2f}'.format(bestK, bestF1, best_alpha))
        #plt.show()

    elif flags[flag] == 'Play':
        # play with data

        df = pd.read_csv('C:\\temp\\deleteme\\Cost_function_all_1_beginning.csv',
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])
        fig, ax = plt.subplots(1, 2)
        K, cDET = detectionErrorTradeOff(df, 'test', cMiss=1.0, cFA=0.1, plt_ax=ax) #, maxK=maxK)
        plt.show()

        df = pd.read_csv('C:\\temp\\deleteme\\Cost_function_all_1_end.csv',
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp", "created_at", "tweet_text"])
        fig, ax = plt.subplots(1, 2)
        K, cDET = detectionErrorTradeOff(df, 'test', cMiss=1.0, cFA=0.1, plt_ax=ax) #, maxK=maxK)
        plt.show()

        df = pd.read_csv('C:\\temp\\deleteme\\average_precision_q2.csv',  # test_precision_recall.csv', #
                         header=None,
                         names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "class", "timestamp",
                                "created_at", "tweet_text"])
        process_data(df)