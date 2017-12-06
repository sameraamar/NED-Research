

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

import os
import sys
import rank_metrics as rm
from sklearn.metrics import average_precision_score, precision_recall_curve, \
                            roc_curve, roc_auc_score, auc, confusion_matrix, \
                            precision_recall_fscore_support

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rank_metrics import ndcg_at_k, dcg_at_k
from scipy import interp
import random

NUMBER_OF_K = 100 #in fact we have 370 positive points only
counter = 0
debug = False
BASE_FOLDER = 'C:/temp/results/cluster_9 - deleteme/' #C:/temp/results/cluster_9_no_timestamp/ml/'
INPUT_FOLDER = BASE_FOLDER + '/ml/' #C:/temp/results/cluster_9_no_timestamp/ml/'
OUTPUT_FOLDER = BASE_FOLDER + '/out/'
ml_files_only = True
console_mode = False
Cmiss = 40.0
Cfa = 0.1

def plot(plt):
    global counter
    #plt.show()

    plt.savefig(OUTPUT_FOLDER + '/{0:04d}.jpg'.format(counter), dpi = 200)
    plt.close()
    counter+=1

def calc_average_precision(y_test, y_scores):
    average_precision = average_precision_score(y_test, y_scores)
    return average_precision

def calc_roc_auc_score(y_true, y_scores, pos_label=1):
    res_roc_auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    return res_roc_auc_score, fpr, tpr, thresholds, roc_auc

# taken from https://github.com/lefterav/rankeval/tree/master/src/evaluation/ranking
def _calculate_gains(predicted_rank_vector, original_rank_vector, verbose=False):
    """
    Calculate the gain for each one of the predicted ranks
    @param predicted_rank_vector: list of integers representing the predicted ranks
    @type predicted_rank_vector: [int, ...]
    @param original_rank_vector: list of integers containing the original ranks
    @type original_rank_vector: [int, ...]
    @return: a list of gains, relevant to the DCG calculation
    @rtype: [float, ...]
    """

    # the original calculation
    r = predicted_rank_vector
    n = len(original_rank_vector)

    # the relevance of each rank is inv proportional to its rank index
    l = [n - i + 1 for i in original_rank_vector]

    expn = 2 ** n
    gains = [0] * n

    # added this line to get high gain for lower rank values
    #    r = r[::-1]
    for j in range(n):
        gains[r[j] - 1] = (2 ** l[j] - 1.0) / expn

        print("j={}\nr[j]={}\nl[j]={}\n".format(j, r[j], l[j]))
        print("gains[{}] = ".format(r[j] - 1))
        print("\t(2**l[j]-1.0) / 2**n =")
        print("\t(2**{}-1.0) / 2**{}=".format(l[j], n))
        print("\t{} / {} =".format((2 ** l[j] - 1.0), expn))
        print("{}".format((2 ** l[j] - 1.0) / expn))
        print("gains = {}".format(gains))

    assert min(gains) >= 0, 'Not all ranks present'
    return gains


def idcg(gains, k):
    """
    Calculate the Ideal Discounted Cumulative Gain, for the given vector of ranking gains
    @param gains: a list of integers pointing to the ranks
    @type gains: [float, ...]
    @param k: the DCG cut-off
    @type k: int
    @return: the calculated Ideal Discounted Cumulative Gain
    @rtype: float
    """
    # put the ranks in an order
    gains.sort()
    # invert the list
    gains = gains[::-1]
    ideal_dcg = sum([g / log(j + 2) for (j, g) in enumerate(gains[:k])])
    return ideal_dcg


def ndgc_err(predicted_rank_vector, original_rank_vector, k=None):
    """
    Calculate the normalize Discounted Cumulative Gain and the Expected Reciprocal Rank on a sentence level
    This follows the definition of U{DCG<http://en.wikipedia.org/wiki/Discounted_cumulative_gain#Cumulative_Gain>}
    and U{ERR<http://research.yahoo.com/files/err.pdf>}, and the implementation of
    U{Yahoo Learning to Rank challenge<http://learningtorankchallenge.yahoo.com/evaluate.py.txt>}
    @param predicted_rank_vector: list of integers representing the predicted ranks
    @type predicted_rank_vector: [int, ...]
    @param original_rank_vector: list of integers containing the original ranks.
    @type original_rank_vector: [int, ...]
    @param k: the cut-off for the calculation of the gains. If not specified, the length of the ranking is used
    @type k: int
    @return: a tuple containing the values for the two metrics
    @rtype: tuple(float,float)
    """
    # Number of documents

    r = predicted_rank_vector.normalize(ties='ceiling').integers()
    l = original_rank_vector.normalize(ties='ceiling').integers()
    n = len(l)

    # if user doesn't specify k, set equal to the ranking length
    if not k:
        k = n

    # make sure that the lists have the right dimensions
    assert len(r) == n, 'Expected {} ranks, but got {}.'.format(n, len(r))
    gains = _calculate_gains(r, l)

    # ERR calculations
    p = 1.0
    err = 0.0
    for j in range(n):
        r = gains[j]
        err += p * r / (j + 1.0)
        p *= 1 - r
    4
    # DCG calculation
    dcg = sum([g / log(j + 2) for (j, g) in enumerate(gains[:k])])

    # NDCG calculation
    ideal_dcg = idcg(gains, k)

    if ideal_dcg:
        ndcg = dcg / ideal_dcg
    else:
        ndcg = 1.0

    return ndcg, err


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
    fullpath = path + '/' + filename
    df = read_csv(fullpath)

    df_sorted = df.sort_values(['y_scores'], ascending=[False])
    # print(df_sorted[['y_true', 'y_pred', 'y_scores']].head(10))

    y_true = df_sorted['y_true']
    y_scores = df_sorted['y_scores']
    y_pred = df_sorted['y_pred']

    return y_scores, y_true, y_pred

def classification_results(files):
    for filename in files:
        try:
            y_scores, y_true, y_pred = prepare_dataset(path, filename)
        except Exception as e:
            print('Error:' ,e)
            continue

        yield (y_scores, y_true, y_pred, filename.replace('.csv', ''))

def petrovic_entropy_users(path, filename):
    fullpath = path + '/' + filename
    df = read_csv(fullpath)

    alpha_values = np.linspace(0.0, 1.0, 10, dtype=float)
    for alpha in alpha_values: #np.line.arange(0.0, 1.1, 0.1):
        df_sorted = sort_data(df, alpha)

        y_true = df_sorted['y_true']
        y_scores = df_sorted['y_scores']
        y_pred = None #df_sorted['y_pred']
        yield y_scores, y_true, y_pred, 'Alpha={0:.2f}'.format(alpha)

def petrovic_fast_growing(path, filename):
    fullpath = path + '/' + filename
    df = read_csv(fullpath)

    for sortBy in ["p_20k", "p_40k", "p_60k", "p_80k"]:
        df_sorted = df.sort_values([sortBy], ascending=[False])

        y_true = df_sorted['y_true']
        y_scores = df_sorted[sortBy]
        y_pred = None  # df_sorted['y_pred']
        yield y_scores, y_true, y_pred, 'SortBy={0}'.format(sortBy)


def petrovic_fast_growing_entropy_user(path, filename):
    fullpath = path + '/' + filename
    df = read_csv(fullpath)

    r = df_sorted.max()-df_sorted.min()
    r = r / 1000 # create 1000 bins
    df_sorted
    df_sorted = df.sort_values(["p_20k"], ascending=[False])

    alpha_values = np.linspace(0.0, 1.0, 10, dtype=float)
    for alpha in alpha_values: #np.line.arange(0.0, 1.1, 0.1):
        df_sorted = sort_data(df, alpha)

        y_true = df_sorted['y_true']
        y_scores = df_sorted['y_scores']
        y_pred = None #df_sorted['y_pred']
        yield y_scores, y_true, y_pred, 'Alpha={0:.2f}'.format(alpha)


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
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right", prop={'size': 6})
    plot(plt)

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
    plt.legend(prop={'size': 6})
    plt.title('Best: {0} (AP={1:.3f})'.format(bestTitle, bestAP), fontsize=10)
    plt.suptitle('2-class Precision-Recall curve', fontsize=18)

    plot(plt)

def my_average_precision(datasets):
    #ap = []
    #x = []

    file = open(OUTPUT_FOLDER + '/average_precision.csv', '+w')
    lines = ['Algorithm\tAP\n']
    for y_scores, y_true, y_pred, title in datasets:
        average_precision = calc_average_precision(y_true, y_scores)
        #print('{0}: Average Precision score: {1:0.7f}'.format(title, average_precision))
        #ap.append( average_precision )
        #x.append( title )
        lines.append('{0}\t{1:.7f}\n'.format(title, average_precision))

    file.writelines(lines)
    file.close()
    # fig, ax = plt.subplots(1, 1)
    # x_int = range(len(ap))
    # ax.bar(x_int, ap)
    # plt.xticks(x_int, x, rotation=10, horizontalalignment="right", fontsize=7)
    # #plt.margins(1.0)
    # plt.title('Average Precision')

    #vals = ax.get_xticks()
    #ax.set_xticklabels([x[int(i)-1] for i in vals])

    # plot(plt)


def over_k(y_true, max_number_of_k, random_false_indices=True, uniform=True):
    r = np.asarray(y_true) != 0
    z = r.nonzero()[0]
    if not z.size:
        return None
    first = z[0]
    last = z[z.size - 1]

    flag = True
    if not flag:
        k_values = random.sample(range(1, len(y_true)+1), max_number_of_k)
        k_values = np.concatenate(( k_values , z ) )
        k_values = set(k_values)
        k_values = list(k_values)
        k_values.sort()

        for k in k_values:
            yield k

    if flag:
        if z.size > max_number_of_k:

            if uniform:
                z = random.sample(set(z), max_number_of_k)
            else:
                probability_distribution = np.asarray( [pow(0.9, i) for i in np.arange(0, z.size, 1)] )
                w = probability_distribution / np.sum(probability_distribution)

                #probability_distribution = np.asarray( [z.size-i for i in np.arange(0, z.size, 1)] )
                #w = probability_distribution / np.sum(probability_distribution)

                # i = np.asarray([z.size-1]*z.size) - np.arange(z.size)
                # w = np.exp(i / 10.)  # higher weights for smaller index values
                # w /= w.sum()  # weight must be normalized
                z = np.random.choice(z, size=max_number_of_k, replace=False, p=w)

            z = np.sort(z)
            z = np.concatenate([[first], z, [last]], axis=0) #force adding first and last items

        additional = np.linspace(last+1, len(y_true)-1, 100, dtype=int)
        additional = set(additional)
        additional = list(additional)
        additional.sort()

        z = np.concatenate([[1], z, additional], axis=0) #force adding first and last items

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


def my_det_score(datasets, draw=False):
    global NUMBER_OF_K
    max_number_of_k = NUMBER_OF_K

    if draw:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(20, 12)

    file = open(OUTPUT_FOLDER + '/det.csv', "+w")
    lines = ['Algorithm\tK\tTP\tBest Cdet\tBest Cdet(normalized)\tEvent count']
    for y_scores, y_true, y_pred, title in datasets:
        kList = []
        CList = []
        C_normList = []
        tp_list = []
        true_count = y_true.sum()
        Ptarget = true_count / y_true.count()
        Pnon_target = 1.0 - Ptarget

        print('{0}: P(target) = {1:.5f}'.format(title, Ptarget))

        normalize_with_me = np.min([Cmiss * Ptarget , Cfa * Pnon_target])

        print('Calculating Cdet for {0}'.format(title))

        if debug:
            print("k\ttp\tfp\ttn\tfn\tP(miss)\tP(FA)\tCdet\tCnorm")

        for k in over_k(y_true, max_number_of_k):
            if(debug and len(CList) % int(10) == 0):
                print('handled', len(CList), 'k values.')

            y_pred = np.concatenate( ([1]*k, [0]*(len(y_true)-k)), axis=0)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=None).ravel()

            Pmiss = 1.0 * fn / (fn + tp)
            Pfa = 1.0 * fp / (fp + tn)

            Cdet = Cmiss * Pmiss * Ptarget + Cfa * Pfa * Pnon_target

            Cnorm = Cdet / normalize_with_me

            if debug:
                print("{0}\t{1:.10f}\t{2:.10f}\t{3:.10f}\t{4:.10f}\t{5:.10f}\t{6:.10f}\t{7:.10f}\t{8:.10f}".format(k, tp, fp, tn, fn, Pmiss, Pfa, Cdet, Cnorm))

            CList.append(Cdet)
            C_normList.append(Cnorm)
            tp_list.append(tp)
            kList.append(k)

        idx = np.argmin(CList)
        summary = "{0}\t{1}\t{2}\t{3:0.5f}".format(title, kList[idx], tp_list[idx], CList[idx])

        if draw:
            ax[0].scatter(kList, CList)
            ax[0].plot(kList, CList, '-x', label="DET ({0}) - {1:0.5f} @ {2}".format(title, CList[idx], kList[idx]))
            ax[0].legend(loc="upper right", prop={'size': 6})

            ax[0].set_title("DET@k")
            ax[0].set_xlabel("k")
            ax[0].set_ylabel("Score ([0,1])")

        idx = np.argmin(CList)
        summary = summary + "\t{0}\t{1}\t{2:0.5f}\t{3}".format(kList[idx], tp_list[idx], C_normList[idx], true_count)

        if draw:
            ax[1].scatter(kList, C_normList)
            ax[1].plot(kList, C_normList, '-o', label="normDET ({0}) - {1:0.5f} @ {2}".format(title, C_normList[idx], kList[idx]))
            ax[1].legend(loc="upper right", prop={'size': 6})

            ax[1].set_title("normalized DET@k")
            ax[1].set_xlabel("k")
            ax[1].set_ylabel("Score ([0,1])")

        lines.append(summary)

    file.writelines('\n'.join(lines))
    file.close()

    if draw:
        #plt.annotate('Cmiss={0:.2f}, Cfa={0:.2f}'.format(Cmiss, Cfa))
        plt.suptitle('Detection Error Tradeoff cost function (regular vs. normalized)\n' + 'Cmiss={0:.2f}, Cfa={1:.2f}'.format(Cmiss, Cfa))
        plt.legend(prop={'size': 6})

        plot(plt)

def my_precision_recall_f1(datasets, measure):
    modes = ['binary', 'micro', 'weighted', 'macro']
    modes = ['binary', 'macro']
    global NUMBER_OF_K
    max_k  = NUMBER_OF_K
    for y_scores, y_true, y_pred, title in datasets:
        kList = []
        p = {}
        r = {}
        f = {}
        s = {}
        for k in over_k(y_true, max_k):
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

        for m in modes:
            fig, ax = plt.subplots(1, 1)
            #ax = plt.subplot()
            fig.set_size_inches(20, 12)

            print(title, end='')
            if 0 in measure:
                __plot_curve(ax, kList, p[m], 'Precision')

            if 1 in measure:
                __plot_curve(ax, kList, r[m], 'Recall')

            if 2 in measure:
                __plot_curve(ax, kList, f[m], 'F1')

            ax.legend(loc="upper right", prop={'size': 6})
            ax.set_title("Score@k (" + title + ") - " + m)
            ax.set_xlabel("k")
            ax.set_ylabel("Score ([0,1])")

            plot(plt)

def my_ndcg(datasets):
    global NUMBER_OF_K
    max_k = NUMBER_OF_K
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 12)

    for y_scores, y_true, y_pred, title in datasets:

        ndcg_values = []
        dcg_values = []
        kList = []
        for k in over_k(y_true, max_k):
            tmp = ndcg_at_k(y_true, k, method=1)
            ndcg_values.append( tmp )

            tmp = dcg_at_k(y_true, k, method=1)
            dcg_values.append( tmp )

            kList.append( k )

        __plot_curve(ax[0], kList, ndcg_values, "nDCG" + title)
        __plot_curve(ax[1], kList, dcg_values, "DCG" + title)

        ax[0].legend(loc="upper right", prop={'size': 6})
        ax[0].set_title("nDCG@k (" + title + ")")
        ax[0].set_xlabel("k")
        ax[0].set_ylabel("nDCG ([0,1])")

        ax[1].legend(loc="upper right", prop={'size': 6})
        ax[1].set_title("DCG@k (" + title + ")")
        ax[1].set_xlabel("k")
        ax[1].set_ylabel("DCG")

    plt.suptitle('DCG / nDCG ' + title)
    plot(plt)

def __plot_curve(ax, x, y, name):
    ax.scatter(x, y)
    ax.plot(x, y, label=name + " (" + m + ")")
    i = np.argmax(y)
    print('Best '+name+' is {0:.4f} at {1}'.format(y[i], x[i]))

if __name__ == "__main__":

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not console_mode:
        orig_stdout = sys.stdout
        f = open(OUTPUT_FOLDER + '/out.txt', 'w')
        sys.stdout = f

    # print(ndcg_at_k([0], 5, method=1))
    # print(ndcg_at_k([1], 5, method=1))
    # print(ndcg_at_k([1, 0], 5, method=1))
    # print(ndcg_at_k([0, 1], 5, method=1))
    # print(ndcg_at_k([0, 1, 1], 5, method=1))
    # print(ndcg_at_k([0, 1, 1, 1], 5, method=1))
    #
    # rank = [2, 2, 1, 0]
    # print('NGDC: ', ndcg_at_k(rank, 4, method=0))
    # print('NGDC: ', ndcg_at_k([2, 2, 1, 0], 4, method=1))
    # print('NGDC: ', ndcg_at_k([2, 1, 2, 0], 4, method=1))
    # print('NGDC: ', ndcg_at_k([2, 1, 2, 0], 4, method=0))

    path = INPUT_FOLDER

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.csv')]

    PRECISION_RECALL_CURVE = "Precision-Recall Curve"
    AVERAGE_PRECISION = "Average Precision"
    ROC_AUC_CURVE = "ROC-AUC"
    DET_SCORE_FUNCTION = "DET"
    PRECISION_RECALL_F1 = "F1"
    PRECISION_AT_K = "Precision@K"
    NDCG = "nDCG"

    metrics = [DET_SCORE_FUNCTION, PRECISION_RECALL_CURVE, AVERAGE_PRECISION, ROC_AUC_CURVE, PRECISION_RECALL_F1, PRECISION_AT_K, NDCG]
    metrics = [DET_SCORE_FUNCTION, PRECISION_RECALL_CURVE, AVERAGE_PRECISION, PRECISION_RECALL_F1, PRECISION_AT_K, NDCG]
    metrics = [AVERAGE_PRECISION, DET_SCORE_FUNCTION]

    print('Metric: ', metrics)
    for m in metrics:
        if NDCG == m:
            if not ml_files_only:
                my_ndcg( petrovic_fast_growing(path, files[0]) )
                my_ndcg( petrovic_entropy_users(path, files[0]) )
            my_ndcg( classification_results(files) )

        if PRECISION_RECALL_F1 == m or PRECISION_AT_K == m:
            measure = [0, 1, 2]
            if m == PRECISION_AT_K:
                measure = [0]
            if not ml_files_only:
                my_precision_recall_f1( petrovic_fast_growing(path, files[0]), measure )
                my_precision_recall_f1( petrovic_entropy_users(path, files[0]), measure )
            my_precision_recall_f1( classification_results(files), measure )

        if DET_SCORE_FUNCTION == m:
            if not ml_files_only:
                my_det_score( petrovic_fast_growing(path, files[0]) )
                my_det_score( petrovic_entropy_users(path, files[0]) )
            my_det_score( classification_results(files) )

        if PRECISION_RECALL_CURVE == m:
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
            if not ml_files_only:
                my_precision_recall_curve( petrovic_fast_growing(path, files[0]) )
                my_precision_recall_curve( petrovic_entropy_users(path, files[0]) )
            my_precision_recall_curve( classification_results(files) )

        if ROC_AUC_CURVE == m:
            print('The area under the ROC curve (AUC) is a commonly used measure which characterizes the trade-off between true positives and false positives as '
                  'a threshold parameter is varied. In our case, the parameter corresponds to the number of items returned (or, predicted as relevant). '
                  'AUC can equivalently be calculated by counting the portion of incorrectly ordered pairs (i.e., j ≺y i, i relevant and j irrelevant), and subtracting '
                  'from 1. This formulation leads to a simple and efficient separation oracle, described by Joachims (2005). '
                  'Note that AUC is position-independent: an incorrect pair-wise ordering at the bottom of the list impacts the score just as much as an error at the top of the list. '
                  'In effect, AUC is a global measure of list-wise cohesion. (https://bmcfee.github.io/papers/mlr.pdf)')

            if not ml_files_only:
                my_roc_curve( petrovic_fast_growing(path, files[0]) )
                my_roc_curve( petrovic_entropy_users(path, files[0]) )
            my_roc_curve( classification_results(files) )

        if AVERAGE_PRECISION == m:
            if not ml_files_only:
                my_average_precision( petrovic_fast_growing(path, files[0]) )
                my_average_precision( petrovic_entropy_users(path, files[0]) )
            my_average_precision( classification_results(files) )

    if not console_mode:
        sys.stdout = orig_stdout
        f.close()

