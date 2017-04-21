from sklearn.datasets import load_iris
import pandas as pd
from pandasql import sqldf
from pandasql import load_meat, load_births
import re

births = load_births()
meat = load_meat()
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
iris_df.columns = [re.sub("[() ]", "", col) for col in iris_df.columns]

dataset = pd.read_csv('c:/temp/data-small.txt', sep='\t')

print(sqldf("SELECT * FROM iris_df LIMIT 10;", locals()))
print(sqldf("SELECT sepalwidthcm, species FROM iris_df LIMIT 10;", locals()))

print(sqldf("SELECT * FROM dataset LIMIT 10;", locals()))