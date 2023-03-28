import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
data = pd.read_csv(r'D:\Datasets\water_potability.csv')
data


data.isnull().sum()

data.describe()

data.fillna(data.mean(),inplace=True)
data

X = data.drop('Potability', axis=1)
Y = data['Potability']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, shuffle=True, random_state=0)
X_train

Y_train

X_test

Y_test



from sklearn.metrics import accuracy_score, confusion_matrix



Y_test.shape

from sklearn import metrics



from sklearn.ensemble import ExtraTreesClassifier
model_etc=ExtraTreesClassifier(n_estimators=900, random_state=1)
model_etc.fit(X_train,Y_train)
prediction_etc=model_etc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(Y_test, prediction_etc)* 100, '%')


pickle.dump(model_etc,open("model.pkl","wb"))









