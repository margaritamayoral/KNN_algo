from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
#%matpltlib inline

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('input/Iris.csv')
head = data.head(10)
info = data.info()
description = data.describe()
y_counts = data['Species'].value_counts()
print(head)
print(info)
print(description)
print(y_counts)

tmp = data.drop('Id', axis=1)
print(tmp)

g = sns.pairplot(tmp, hue='Species', markers='+')
plt.savefig('output/g.png')
#plt.show(g)

g1 = plt.figure(2)
g1 = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')
plt.savefig('output/g1.png')
#plt.show(g1)

g2 = plt.figure(3)
g2 = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')
plt.savefig('output/g2.png')
g3 = plt.figure(4)
g3 = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')
plt.savefig('output/g3.png')
g4 = plt.figure(5)
g4 = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')
plt.savefig('output/g4.png')

###### Modeling with scikit-learn#######
X=data.drop(['Id', 'Species'], axis=1)
y=data['Species']
print(X.shape)
print(y.shape)

### Training and test on the same dataset ###
## * This method is not sugested since the end goal is to predict iris species using a dataset the model has ever seen before ##
## * There is also the risk of overfitting the training data ##

# experimenting with different n values ###

k_range = list(range(1, 26))
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    scores.append(metrics.accuracy_score(y, y_pred))
plt.figure(6)
plt.plot(k_range, scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')
plt.savefig('output/KNN.png')
#plt.show()




