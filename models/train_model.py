import numpy as np
import pandas as pd
import joblib

dataset = pd.read_csv('customer_churn_dataset-testing-master.csv')
x= dataset.iloc[:, 1: -1].values
y= dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 7] = le.fit_transform(x[:, 7])


x[:, 6] = le.fit_transform(x[:, 6])

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x, y , test_size=0.25 , random_state=0)


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=500, random_state=42)
random_forest.fit(x_train, y_train)

joblib.dump(random_forest, 'train_model.pkl')