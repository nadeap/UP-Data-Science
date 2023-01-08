#Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Open Dataset
data = pd.read_csv('ipm.csv')
data.head()

#Melihat ukuran data
data.shape

#Melihat informasi dataset
data.info()

#Melihat Jumlah Nilai Null dalam data
data.isnull().sum()

#Melihat deskriptif statistik dataset
data.describe()

#Melihat jumlah jenis data dalam kolom IPM
data['IPM'].value_counts()

#LabelEncoder Categorical Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['IPM'] = le.fit_transform(data['IPM'])

#Declare Dataset
x = data.drop(['IPM'],axis=True)
y = data['IPM']

#Split Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

#Modelling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#KNN
knn = KNeighborsClassifier() #auto 5
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
score = accuracy_score(y_test, y_pred)
print('Akurasi = ',score)
print('Training set score: {:.4f}'.format(knn.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(knn.score(x_test, y_test)))
from sklearn.metrics import classification_report
confusion_matrix(y_test, y_pred) 
print(classification_report(y_test, y_pred))

#Decision Tree
for max_d in range(1,10):
    model = DecisionTreeClassifier(max_depth=max_d, random_state=42)
    model.fit(x_train, y_train)
    print('Hasil Training model untuk max_depth {} adalah :'.format(max_d), model.score(x_train,y_train))
    print('Hasil Testing model untuk max_depth {} adalah :'.format(max_d), model.score(x_test,y_test))
    print('')

modelGini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
modelGini.fit(x_train, y_train)
y_pred_gini = modelGini.predict(x_test)
cm = confusion_matrix(y_test, y_pred_gini)
print(cm)
accuracy_score(y_test, y_pred_gini)
print('Training set score: {:.4f}'.format(modelGini.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(modelGini.score(x_test, y_test)))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_gini))

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred) 
print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))
print(classification_report(y_test, y_pred))

#Simple Prediction
predict = modelGini.predict(np.array([[14.56,8765,8.34,64.34]]))
predict