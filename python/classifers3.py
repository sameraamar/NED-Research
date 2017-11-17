
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn import metrics
#from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.datasets import load_iris

#import scikitplot.plotters as skplt
import matplotlib.pyplot as plt
from scipy import interp

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

feature_col = []

def getdata1():
    # Create an object called iris with the iris data
    iris = load_iris()

    # Create a dataframe with the four feature variables
    df = pd.DataFrame(iris.data) #, columns=iris.feature_names)

    return df

def getdata():

    df = pd.read_csv('C:\\temp\\Sep14\\all_clusters_by_lead_id_bonus_users.csv') #'C:/temp/data-cluster1.csv') #c:/temp/data_minimal_features.csv')
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    global feature_col
    feature_col = df.columns #!= 'class'
    #feature_col = ['entropy', 'unique_users', 'size', 'class']

    return df[feature_col]

def getXy(df):
    print(df.columns)
    X = df.ix[:, df.columns != 'class']

    d = {'yes': 1, 'no': 0}
    y = df['class'].map(d)

    return X, y

def split_data(df, train_perc = 0.75):
    # Create a new column that for each row, generates a random number between 0 and 1, and
    # if that value is less than or equal to .75, then sets the value of that cell as True
    # and false otherwise. This is a quick and dirty way of randomly assigning some rows to
    # be used as the training data and some as the test data.
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= train_perc

    # Create two new dataframes, one with the training rows, one with the test rows
    train, test = df[df['is_train']==True], df[df['is_train']==False]

    train_X = train.ix[:, feature_col]
    test_X = test.ix[:, feature_col]

    #train_X = train.ix[:, df.columns != 'class']
    #test_X = test.ix[:, df.columns != 'class']

    d = {'yes': 1, 'no': 0}
    train_y = train['class'].map(d)
    test_y = test['class'].map(d)

    return train_X, train_y, test_X, test_y

def plot_roc_curve(y_true, y_probas, title='ROC Curves',
                   curves=('micro', 'macro', 'each_class'),
                   ax=None, figsize=None, cmap='spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities
    From: https://github.com/reiinakano/scikit-plot
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".
        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if 'micro' not in curves and 'macro' not in curves and \
            'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)
    probas = y_probas

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true, probas[:, i],
                                      pos_label=classes[i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = metrics.roc_curve(y_true.ravel(),
                                                  probas.ravel())
    roc_auc[micro_key] = metrics.auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = metrics.auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr[i], tpr[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(classes[i], roc_auc[i]))

    if 'micro' in curves:
        ax.plot(fpr[micro_key], tpr[micro_key],
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[micro_key]),
                color='deeppink', linestyle=':', linewidth=4)

    if 'macro' in curves:
        ax.plot(fpr[macro_key], tpr[macro_key],
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[macro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax


# Bagged Decision Trees for Classification

df = getdata()
X, y = getXy(df)

seed = 7
print('DecisionTreeClassifier')
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
scores = model_selection.cross_val_score(model, X, y, cv=kfold)
label = 'DecisionTreeClassifier'
print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
y_pred_prob = model_selection.cross_val_predict(model, X, y, cv=kfold, method='predict_proba')

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, y_pred)
print ('AUC', metrics.auc(false_positive_rate, true_positive_rate))
print ('ROC_AUC', metrics.roc_auc_score(y, y_pred))
print('F1', metrics.f1_score(y, y_pred))
# Print ROC curve
plt.plot(false_positive_rate, true_positive_rate)
plt.show()
ax = plot_roc_curve(y, y_pred_prob)
plt.show()
#metrics.roc_curve(y, y_pred_prob[:,1])

print('AdaBoost')
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
scores = model_selection.cross_val_score(model, X, y, cv=kfold)
label = 'AdaBoost'
print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
y_pred_prob = model_selection.cross_val_predict(model, X, y, cv=kfold, method='predict_proba')

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, y_pred)
print ('AUC', metrics.auc(false_positive_rate, true_positive_rate))
print ('ROC_AUC', metrics.roc_auc_score(y, y_pred))
print('F1', metrics.f1_score(y, y_pred))
#metrics.roc_curve(y, y_pred_prob[:,1])

print('ExtraTreesClassifier')
seed = 7
num_trees = 100
max_features = None # bring all
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
scores = model_selection.cross_val_score(model, X, y, cv=kfold)
label = 'ExtraTreesClassifier'
print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
y_pred_prob = model_selection.cross_val_predict(model, X, y, cv=kfold, method='predict_proba')

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, y_pred)
print ('AUC', metrics.auc(false_positive_rate, true_positive_rate))
print ('ROC_AUC', metrics.roc_auc_score(y, y_pred))
print('F1', metrics.f1_score(y, y_pred))
#metrics.roc_curve(y, y_pred_prob[:,1])

#voting algorithms
if False:
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    # create the sub models
    estimators = []
    model1 = LogisticRegression()
    estimators.append(('logistic', model1))
    model2 = DecisionTreeClassifier()
    estimators.append(('cart', model2))
    model3 = SVC()
    estimators.append(('svm', model3))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    label = 'voting algorithms'
    print("ROC AUC: %0.2f (+/- %0.2f) [%s]"
              % (scores.mean(), scores.std(), label))
    y_pred = model_selection.cross_val_predict(model, X, y, cv=kfold)
    y_pred_prob = model_selection.cross_val_predict(model, X, y, cv=kfold, method='predict_proba')

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y, y_pred)
    print ('AUC', metrics.auc(false_positive_rate, true_positive_rate))
    print ('ROC_AUC', metrics.roc_auc_score(y, y_pred))
    print('F1', metrics.f1_score(y, y_pred))
    #metrics.roc_curve(y, y_pred_prob[:,1])

