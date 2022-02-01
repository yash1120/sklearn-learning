from sklearn import datasets
import numpy as np 
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,shuffle=False)
print(y_test)