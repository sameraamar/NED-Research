

'''
Precision-Recall Curve...
Example taken from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py

Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced.
In information retrieval, precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned.
The precision-recall curve shows the tradeoff between precision and recall for different threshold.
A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate,
and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision),
as well as returning a majority of all positive results (high recall).

A system with high recall but low precision returns many results, but most of its predicted labels are incorrect when compared to the training labels.
A system with high precision but low recall is just the opposite, returning very few results, but most of its predicted labels are correct when compared to the training labels.
An ideal system with high precision and high recall will return many results, with all results labeled correctly.
'''

import rank_metrics as rm
from sklearn.metrics import average_precision_score, precision_recall_curve, \
                            roc_curve, roc_auc_score, auc, confusion_matrix, \
                            precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interp
import random

def calc_average_precision(y_test, y_scores):
    average_precision = average_precision_score(y_test, y_scores)
    return average_precision

def calc_roc_auc_score(y_true, y_scores, pos_label=1):
    res_roc_auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    return res_roc_auc_score, fpr, tpr, thresholds, roc_auc

#*****************************************  M>A>I>N **************************************************

def zscore_normalize(df, column):
    #print('mean({0}): {1}'.format(df[column].mean(), df[column].std(ddof=0)))
    df[column+'_zscore'] = (df[column] - df[column].mean()) / df[column].std(ddof=0)


def sort_data(df, alpha):
    # normalization
    zscore_normalize(df, 'entropy')
    zscore_normalize(df, 'unique_users')

    df["y_scores"] = df["entropy_zscore"] * alpha + df["unique_users_zscore"] * (1 - alpha)

    return df.sort_values(['y_scores'], ascending=[False])


def read_csv(filename):
    df = pd.read_csv(filename)

    d = {'yes': 1, 'no': 0}
    df['y_true'] = df['class'].map(d)
    df['y_pred'] = df['Prediction (class)'].map(d)
    df['y_scores'] = df['P (class=yes)']

    return df


def prepare_dataset(path, filename):
    fullpath = path + filename
    df = read_csv(fullpath)

    df_sorted = df.sort_values(['y_scores'], ascending=[False])
    # print(df_sorted[['y_true', 'y_pred', 'y_scores']].head(10))

    y_true = df_sorted['y_true']
    y_scores = df_sorted['y_scores']
    y_pred = df_sorted['y_pred']
    return y_scores, y_true, y_pred

def classification_results(files):
    for filename in files:
        y_scores, y_true, y_pred = prepare_dataset(path, filename)
        yield (y_scores, y_true, y_pred, filename)

def petrovic_entropy_users(path, filename):
    fullpath = path + filename
    df = read_csv(fullpath)

    alpha_values = np.linspace(0.0, 1.0, 10, dtype=float)
    for alpha in alpha_values: #np.line.arange(0.0, 1.1, 0.1):
        df_sorted = sort_data(df, alpha)

        y_true = df_sorted['y_true']
        y_scores = df_sorted['y_scores']
        y_pred = None #df_sorted['y_pred']
        yield y_scores, y_true, y_pred, 'Alpha={0:.2f}'.format(alpha)

def petrovic_fast_growing(path, filename):
    fullpath = path + filename
    df = read_csv(fullpath)

    for sortBy in ["p_20k", "p_40k", "p_40k", "p_80k"]:
        df_sorted = df.sort_values([sortBy], ascending=[False])

        y_true = df_sorted['y_true']
        y_scores = df_sorted[sortBy]
        y_pred = None  # df_sorted['y_pred']
        yield y_scores, y_true, y_pred, 'SortBy={0}'.format(sortBy)


def my_roc_curve(datasets):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    for (y_scores, y_true, y_pred, title) in datasets:
        res_roc_auc_score, fpr, tpr, thresholds, roc_auc = calc_roc_auc_score(y_true, y_scores)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC[{0}] (AUC = {1:.2f})'.format(title, roc_auc))
        print(ROC_AUC_CURVE, title, res_roc_auc_score)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def my_precision_recall_curve(datasets):
    bestAP = 0
    bestTitle = ''
    for y_scores, y_true, y_pred, title in datasets:
        average_precision = average_precision_score(y_true, y_scores)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        if average_precision > bestAP:
            bestAP = average_precision
            bestTitle = title

        axis = '{0} - AP={1:0.7f}'.format(title, average_precision)

        plt.step(recall, precision, alpha=0.2, where='post', label=axis)
        plt.fill_between(recall, precision, step='post', alpha=0.2)
        plt.xlabel('Purchase amount', fontsize=18)
        #plt.title('{0}\n2-class Precision-Recall curve: AP={1:0.7f}'.format(title, average_precision))

        #average_precision = calc_precision_recall_curve(y_true, y_scores, title)
        print('File ', title, ': precision=', rm.r_precision(y_true), ' Avergae Precision: ', average_precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend()
    plt.title('Best: {0} (AP={1:.3f})'.format(bestTitle, bestAP), fontsize=10)
    plt.suptitle('2-class Precision-Recall curve', fontsize=18)

    plt.show()

def my_average_precision(datasets):
    for y_scores, y_true, y_pred, title in datasets:
        average_precision = calc_average_precision(y_true, y_scores)
        print('{0}: Average Precision score: {1:0.7f}'.format(title, average_precision))

def over_k(y_true, max_k, random_false_indices=True):
    r = np.asarray(y_true) != 0
    z = r.nonzero()[0]
    if not z.size:
        return None

    if z.size > max_k:
        first = z[0]
        last = z[z.size - 1]
        z = random.sample(set(z), max_k)
        z = np.sort(z)
        z = np.concatenate([[first], z, [last]], axis=0)

    previousK = None
    for k in z:
        small_list = [k]
        if not previousK is None:
            k1 = (k+previousK)/2
            k1 = int(k1)
            if k1<k and k1>previousK:
                small_list = [k1, k]

        if random_false_indices:
            previousK = k
        # end of trick

        for k in small_list:
            yield k


def my_det_score(datasets):
    Cmiss = 1.0
    Cfa = 0.1
    max_k = 10
    fig, ax = plt.subplots(1, 2)

    for y_scores, y_true, y_pred, title in datasets:

        kList = []
        CList = []
        C_normList = []
        Ptarget = y_true.sum() / y_true.count()
        Pnon_target = 1.0 - Ptarget

        nnn = np.min([Cmiss * Ptarget , Cfa * Pnon_target])

        print('Calculating Cdet for {0}'.format(title))

        for k in over_k(y_true, max_k):
            if(len(CList) % int(10) == 0):
                print('handled', len(CList), 'k values.')

            y_pred = np.concatenate( ([1]*k, [0]*(len(y_true)-k)), axis=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, 0], sample_weight=None).ravel()

            Pmiss = 1.0 * fn / (fn + tp)
            Pfa = 1.0 * fp / (fp + tn)

            Cdet = Cmiss * Pmiss * Ptarget + Cfa * Pfa * Pnon_target

            Cnorm = Cdet / nnn

            CList.append(Cdet)
            C_normList.append(Cnorm)
            kList.append(k)


        ax[0].scatter(kList, CList)
        ax[0].plot(kList, CList, '-x', label="DET ({0}) - {1:0.5f}".format(title, np.min(CList)))
        ax[0].legend(loc="upper right")

        ax[0].set_title("DET@k")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("Score ([0,1])")

        ax[1].scatter(kList, C_normList)
        ax[1].plot(kList, C_normList, '-o', label="normDET ({0}) - {1:0.5f}".format(title, np.min(C_normList)))
        ax[1].legend(loc="upper right")

        ax[1].set_title("normalized DET@k")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("Score ([0,1])")

    #plt.annotate('Cmiss={0:.2f}, Cfa={0:.2f}'.format(Cmiss, Cfa))
    plt.suptitle('Detection Error Tradeoff cost function (regular vs. normalized)\n' + 'Cmiss={0:.2f}, Cfa={1:.2f}'.format(Cmiss, Cfa))
    plt.legend()

    plt.show()

def my_precision_recall_f1(datasets):
    modes = ['binary', 'micro', 'weighted', 'macro']
    max_k  = 14000
    for y_scores, y_true, y_pred, title in datasets:
        kList = []
        p = {}
        r = {}
        f = {}
        s = {}
        for k in over_k(y_true, max_k, random_false_indices=False):
            y_pred = np.concatenate( ([1]*k, [0]*(len(y_true)-k)), axis=0)

            for m in modes:
                if not m in p:
                    p[m] = []
                if not m in r:
                    r[m] = []
                if not m in f:
                    f[m] = []
                if not m in s:
                    s[m] = []

                precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average=m, pos_label=1)
                p[m].append(precision)
                r[m].append(recall)
                f[m].append(fscore)
                s[m].append(support)

            kList.append(k)

        fx = plt.subplot()
        fx.scatter(kList, p[m])
        fx.plot(kList, p[m], '-x', label="Precision")
        fx.legend(loc="upper right")

        fx.scatter(kList, r[m])
        fx.plot(kList, r[m], '-x', label="Recall")
        fx.legend(loc="upper right")

        fx.set_title("Score@k (" + title + ") - " + m)
        fx.set_xlabel("k")
        fx.set_ylabel("Score ([0,1])")

        plt.show()

if __name__ == "__main__":
    path = 'C:/temp/deleteme/classification.tweet_based_old_timestamp_calc/'

    files = ['adaboost.1.SMOTE.csv', 'bagging_R1.1.SMOTE.csv', 'bagging_dtree.1.SMOTE.csv', 'randomforest.1.SMOTE.csv',
             'adaboost.2.Under-Sampling.csv', 'bagging_R1.2.Under-Sampling.csv', 'bagging_dtree.2.Under-Sampling.csv', 'randomforest.2.Under-Sampling.csv']

    PRECISION_RECALL_CURVE = "Precision-Recall Curve"
    PRECISION_AT_K = "Precision@K"
    AVERAGE_PRECISION = "Average Precision"
    ROC_AUC_CURVE = "ROC-AUC"
    DET_SCORE_FUNCTION = "DET"
    PRECISION_RECALL_F1 = "F1"

    metrics = [PRECISION_RECALL_F1, DET_SCORE_FUNCTION, ROC_AUC_CURVE, AVERAGE_PRECISION, PRECISION_RECALL_CURVE]

    print('Metric: ', metrics)

    if PRECISION_RECALL_F1 in metrics:
        my_precision_recall_f1( petrovic_fast_growing(path, files[0]) )
        my_precision_recall_f1( petrovic_entropy_users(path, files[0]) )
        my_precision_recall_f1( classification_results(files) )

    if DET_SCORE_FUNCTION in metrics:
        my_det_score( petrovic_fast_growing(path, files[0]) )
        my_det_score( petrovic_entropy_users(path, files[0]) )
        my_det_score( classification_results(files) )

    if PRECISION_RECALL_CURVE in metrics:
        print('Precision-Recall is a useful measure of success of prediction when the classes are very imbalanced. In information retrieval, '
              'precision is a measure of result relevancy, while recall is a measure of how many truly relevant results are returned. '
              'The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents '
              'both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false '
              'negative rate. High scores for both show that the classifier is returning accurate results (high precision), '
              'as well as returning a majority of all positive results (high recall). '
              'A system with high recall but low precision returns many results, but most of its predicted labels are incorrect '
              'when compared to the training labels. A system with high precision but low recall is just the opposite, '
              'returning very few results, but most of its predicted labels are correct when compared to the training labels. '
              'An ideal system with high precision and high recall will return many results, with all results labeled correctly.'
              'http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py')
        my_precision_recall_curve( petrovic_fast_growing(path, files[0]) )
        my_precision_recall_curve( petrovic_entropy_users(path, files[0]) )
        my_precision_recall_curve( classification_results(files) )

    if ROC_AUC_CURVE in metrics:
        print('The area under the ROC curve (AUC) is a commonly used measure which characterizes the trade-off between true positives and false positives as '
              'a threshold parameter is varied. In our case, the parameter corresponds to the number of items returned (or, predicted as relevant). '
              'AUC can equivalently be calculated by counting the portion of incorrectly ordered pairs (i.e., j ≺y i, i relevant and j irrelevant), and subtracting '
              'from 1. This formulation leads to a simple and efficient separation oracle, described by Joachims (2005). '
              'Note that AUC is position-independent: an incorrect pair-wise ordering at the bottom of the list impacts the score just as much as an error at the top of the list. '
              'In effect, AUC is a global measure of list-wise cohesion. (https://bmcfee.github.io/papers/mlr.pdf)')

        my_roc_curve( petrovic_fast_growing(path, files[0]) )
        my_roc_curve( petrovic_entropy_users(path, files[0]) )
        my_roc_curve( classification_results(files) )

    if AVERAGE_PRECISION in metrics:
        my_average_precision( petrovic_fast_growing(path, files[0]) )
        my_average_precision( petrovic_entropy_users(path, files[0]) )
        my_average_precision( classification_results(files) )
