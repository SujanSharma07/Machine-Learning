import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pickle


heart = pd.read_csv("heart.csv")



input_x = heart.drop(["target"], axis = 1)
drop = input_x.head(0)
input_x = np.array(input_x)
input_y = heart.drop(drop,axis =1)
input_y = np.array(input_y)
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(input_x, input_y, test_size=0.1)



model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train,y_train)

acc = model.score(x_test, y_test)
print(acc)