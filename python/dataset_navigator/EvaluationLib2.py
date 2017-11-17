# Load pandas
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# Load numpy
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, \
    accuracy_score, auc, roc_curve, average_precision_score
#from rank_metrics import ndcg


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


df = pd.read_csv('C:/ProgramData/MySQL/MySQL Server 5.6/Uploads/tweets_topics2.csv',
                 header=None,
                 names=["tweet_id", "size", "users", "entropy", "rank", "votes", "voters", "is_event", "timestamp", "created_at", "tweet_text"])

#df = df.sort_values('users')
print(df.head())
print(df.count())

#temp = df[(df["users"] == 1) & (df["entropy"] == 1)]

entropies = np.arange(0, 2.5, 0.1)
print(entropies)

precisionList = {}
recallList = {}
F1List = {}

accuracyList = []
roc_aucList = []

modes = ['binary', 'micro', 'macro']
for m in modes:
    precisionList[m] = []
    recallList[m] = []
    F1List[m] = []

# normalization
df["entropy_norm"]=(df["entropy"]-df["entropy"].mean())/df["entropy"].std()
df["users_norm"]=(df["users"]-df["users"].mean())/df["entropy"].std()

print('normalized: ')
print(df["entropy_norm"].head(10))
print(df["users_norm"].head(10))
# min max: normalized_df=(df-df.min())/(df.max()-df.min())

alpha = 0.5
users = 0

for entropy in entropies:
    rank_score = entropy * alpha + users * (1 - alpha)

    #selected = df[df["entropy"] <= entropy]
    df["test"] = (df["entropy_norm"] >= entropy) & (df["users_norm"] >= users)
    d = {True: 1, False: 0}
    predY = df['test'].map(d)
    testY = df['is_event']

    #print(testY.head(), testY.shape)
    #print(predY.head(), predY.shape)

    for m in modes:
        report_lr = precision_recall_fscore_support(testY, predY, average=m, pos_label=1)
        precision = report_lr[0]
        precisionList[m].append(precision)

        recall = report_lr[1]
        recallList[m].append(recall)

        F1List[m].append(report_lr[2])

    accuracy = accuracy_score(testY, predY)
    accuracyList.append(accuracy)

    print ("Score %0.2f: precision = %0.2f, recall = %0.2f, F1 = %0.2f, accuracy = %0.2f" % \
               (entropy, report_lr[0], report_lr[1], report_lr[2], accuracy), end='')

    fpr, tpr, thresholds = roc_curve(testY, predY, pos_label=1)
    auc_area = auc(fpr, tpr)
    roc_aucList.append(auc_area)
    print(", AUC: ", auc_area)

print(precisionList, recallList, F1List, roc_aucList)
print(len(entropies), len(precisionList), len(recallList))

for m in modes:
    plt.scatter(entropies, precisionList[m])
    plt.plot(entropies, precisionList[m], '-o', label="Precision (" + m + ")")

    plt.scatter(entropies, recallList[m])
    plt.plot(entropies, recallList[m], '-x', label="Recall (" + m + ")")

    plt.scatter(entropies, F1List[m])
    plt.plot(entropies, F1List[m], '-x', label="F1 (" + m + ")")

    plt.title("Metrics for entropy field (Precision vs. Recall)")
    plt.xlabel("Entropy")
    plt.xlabel("Score ([0,1])")

    plt.legend()

    plt.show()

plt.scatter(entropies, accuracyList)
plt.plot(entropies, accuracyList, '-.', label="Accuracy")

plt.scatter(entropies, roc_aucList)
plt.plot(entropies, roc_aucList, '--', label="AUC")

plt.title("Metrics for entropy field (Percision vs. Recall)")
plt.xlabel("Entropy")
plt.xlabel("Score ([0,1])")

plt.legend()

plt.show()


#ndcg_values = [ndcg( x for x in entropies]
