import numpy as np
import pandas as pd
import joblib

dataset = pd.read_csv('customer_churn_dataset-testing-master.csv')
x= dataset.iloc[:, 1: -1].values
y= dataset.iloc[:, -1].values

print(x[0:15, 1])
# print(x[0:15, 7])
# print(x[0:15, 6])


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 7] = le.fit_transform(x[:, 7])
x[:, 6] = le.fit_transform(x[:, 6])

from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test = train_test_split(x, y , test_size=0.25 , random_state=0)


from sklearn.preprocessing import RobustScaler
sc= RobustScaler()
x_train=sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(x_train, y_train)

array = np.array([[ 22,0,  25,  14,   4,  27,   0,   0, 598,   9]])
array = sc.transform(array)
print(array)
prediction = random_forest.predict(array)


y_pred = random_forest.predict(x_test)

from sklearn.metrics import accuracy_score

# accuracy_random_forest = random_forest.score(y_pred, y_test)
# print("Accuracy:", accuracy_random_forest)
joblib.dump(random_forest, 'train_model.pkl')
joblib.dump(sc, 'scaler.pkl')
