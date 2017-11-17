
# Load the library with the iris dataset
#from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

def getdata1():
    # Create an object called iris with the iris data
    iris = load_iris()

    # Create a dataframe with the four feature variables
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    return df

feature_col = []

def getdata():
    df = pd.read_csv('C:\\temp\\Sep14\\cluster_by_lead_id_bonus_users.csv') #'C:/temp/data-cluster1.csv') #c:/temp/data_minimal_features.csv')
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    global feature_col

    #feature_col = ['entropy', 'unique_users', 'size']
    feature_col = df.columns != 'class'

    return df

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


def ensemble(df):
    le = LabelEncoder()
    X = df.ix[:, feature_col].values
    Y = le.fit_transform(df.ix[:, 'class'].values)

    numFolds = 5

    kf = KFold(numFolds, shuffle=True)
    conv_X = pd.get_dummies(df.ix[:, feature_col])

    # These are "Class objects". For each Class, find the AUC through
    # 10 fold cross validation.
    Models = [LogisticRegression, RandomForestClassifier] #, SVC]
    params = [{}, {}, {"criterion":'entropy', "probability": True}]
    for Model, param in zip(Models, params):
        total = 0
        acc = 0
        itr = 0
        for train_indices, test_indices in kf.split(X):
            #print('iteration: ', itr)
            itr += 1
            # Get the dataset; this is the way to access values in a pandas DataFrame
            train_X = conv_X.ix[train_indices, :]; train_Y = Y[train_indices]
            test_X = conv_X.ix[test_indices, :]; test_Y = Y[test_indices]

            # Train the model, and evaluate it
            reg = Model(**param)
            reg.fit(train_X, train_Y)
            predictions = reg.predict_proba(test_X)[:, 1]
            predictions2 = reg.predict(test_X)
            fpr, tpr, _ = roc_curve(test_Y, predictions)
            total += auc(fpr, tpr)
            acc += accuracy_score(test_Y, predictions2)

        accuracy = total / numFolds
        acc = acc / numFolds
        print ("{0}: AUC={1}, ACC={2}".format(Model.__name__, accuracy, acc))

# Set random seed
np.random.seed(0)

df = getdata()

ensemble(df)

train_X, train_y, test_X, test_y = split_data(df)

# View the top 5 rows
print(train_X.head())

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train_X))
print('Number of observations in the test data:',len(test_X))

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train_X, train_y)

# Apply the Classifier we trained to the test data (which, remember, it has never seen before)
pred_y = clf.predict(test_X)


# Create confusion matrix
conf_matrix = pd.crosstab(test_y, pred_y)

# View a list of the features and their importance scores
feature_importance = list(zip(train_X, clf.feature_importances_))

print(conf_matrix , feature_importance, roc_auc_score(test_y, pred_y))

