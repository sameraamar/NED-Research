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

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def Cscore(y_true, y_pred):
    CM = confusion_matrix(testY, predY, labels=[0, 1])
    [[tn, fp], [fn, tp]] = CM

    pMiss = fn / (fn + tp)  # miss rate (1-recall)
    pFA = fp / (fp + tn)  # false alarm rate

    cMiss = 1.0
    cFA = 0.1

    cDET = cMiss * pMiss * pTarget + cFA * pFA * pNonTarget
    return cDET

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=64, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics={'output_a': 'accuracy', 'output_b': 'precision', 'output_c': metrics.MAPE})
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', metrics.MAPE])
	return model

#df = pd.read_csv('c:/temp/deleteme/data_clusters_by_bonus_users_tweet_id.csv', sep='\t',quoting=csv.QUOTE_ALL)
df = pd.read_csv('c:/temp/results/cluster_X/cluster_by_tweet_id_with_dwelling.updated.csv', sep='\t',quoting=csv.QUOTE_ALL)

print(df.head(5))
print(list(df.columns.values))

d = {'yes': 1, 'no': 0}
df['class'] = df['class'].map(d)

y = df['class'].values
X = df.loc[:, df.columns != 'class'].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

print('X: ', X.shape)

#X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                stratify=y,
#
#                                                 test_size=0.25)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

class_weight = {0 : 1., 1: 50.}

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True,  random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold, fit_params={'sample_weight': class_weight})
print(results)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

#y_train = pd.DataFrame(y_train, columns = ["class"])
#print(y_train.head(5))

# print(Counter(y_train))
# print(Counter(y_test))
#
# model = Sequential()
#
# print(X_train.shape)
# model.add(Dense(units=64, activation='relu', input_shape=X_train.shape))
# #model.add(Dense(units=10, activation='softmax'))
#
#
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])
#
#
# # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
# model.fit(X_train, y_train, epochs=5, batch_size=32, class_weight=class_weight)
#
# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
# #classes = model.predict(x_test, batch_size=128)
