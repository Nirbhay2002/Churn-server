import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate

dataset = pd.read_csv('customer_churn_dataset-testing-master.csv')
x= dataset.iloc[:, 1: -1].values
y= dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 7] = le.fit_transform(x[:, 7])

# print(x.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [6])], remainder='passthrough')
# x= np.array(ct.fit_transform(x))

x[:, 6] = le.fit_transform(x[:, 6])

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x, y , test_size=0.25 , random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc1 = StandardScaler()
# sc2 = StandardScaler()

# # print(x_train)

# X_train = sc1.fit_transform(x_train)
# X_test = sc2.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
random_forest.fit(x_train, y_train)
# print(X_train)
print(tabulate(x_train[0:7, :], headers='keys', tablefmt='psql'))

accuracy_random_forest = random_forest.score(x_test, y_test)
print(accuracy_random_forest)

joblib.dump(random_forest, 'train_model.pkl')