from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
from collections import Counter

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


# baseline model
def create_baseline(activation_func='relu'):
	# create model
    # Initialize the constructor
    model = Sequential()

    # Add an input layer
    model.add(Dense(200, activation='relu', input_shape=(63,)))

    # Add one hidden layer
    model.add(Dense(100, activation='relu'))

    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))

    return model

#df = pd.read_csv('c:/temp/deleteme/data_clusters_by_bonus_users_tweet_id.csv', sep='\t',quoting=csv.QUOTE_ALL)
df = pd.read_csv('c:/temp/results/cluster_X/cluster_by_tweet_id_with_dwelling.updated.csv', sep='\t',quoting=csv.QUOTE_ALL)

print(df.head(5))
print(list(df.columns.values))

print(df.shape)

#input("press any key")

d = {'yes': 1, 'no': 0}
df['class'] = df['class'].map(d)

print(df['class'].value_counts())
df = df.sort_values(['class'], ascending=[False])
df = df.head(800)
print(df['class'].value_counts())

y = df['class'].values
X = df.loc[:, df.columns != 'class'].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

print('X: ', X.shape)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)


class_weight = {0 : 1., 1: 50.}

for activation_func in ['relu']:
    print('activation function: ' , activation_func)
    model = create_baseline(activation_func=activation_func)

    # Model output shape
    print('model: ', model)

    # Model config
    print('config' , model.get_config())

    # List all weight tensors
    print('get weights: ', model.get_weights())

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
                  #metrics=['top_k_categorical_accuracy'])

    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0) #, class_weight=class_weight)

    y_pred = model.predict(X_test)
    print('y_test classes: ', len(y_test), y_test)
    print('y_pred classes: ', len(y_pred), y_pred)
    score = model.evaluate(X_test, y_test, verbose=1)

    print('Model Score: ' , score)

    # Confusion matrix
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))

    # Precision
    print('Precision score', precision_score(y_test, y_pred))

    # Recall
    print('Recall score', recall_score(y_test, y_pred))

    # F1 score
    print('F1 score', f1_score(y_test,y_pred))

    # Cohen's kappa
    print('Cohen kappa: ' , cohen_kappa_score(y_test, y_pred))



