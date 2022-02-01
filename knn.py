import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import neighbors,metrics,datasets
from sklearn.preprocessing import LabelEncoder
data = data=pd.read_csv("datasets\car.data",names=["buying","maint","doors","persons","lug_boot","safety","class"])
x = data[[
    'buying',
    'maint',
    'safety'
]].values
y = data[['class']]

# coversion of data
le = LabelEncoder()
for i in range(len(x[0])):
    x[:,i]=le.fit_transform(x[:,i])

label_maping={
    'acc':0,
    'good':1,
    'unacc':2,
    'vgood':3
}
y['class'] = y["class"].map(label_maping)

knn = neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y)
knn.fit(x_train,y_train)
predict = knn.predict(x_test)
accuracy = metrics.accuracy_score(predict,y_test)
print(" prediction is : ",predict)
print(" accuracy is ; ", accuracy)