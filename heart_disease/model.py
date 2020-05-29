import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle


heart = pd.read_csv("heart.csv")



input_x = heart.drop(["target"], axis = 1)
drop = input_x.head(0)
input_x = np.array(input_x)
input_y = heart.drop(drop,axis =1)
input_y = np.array(input_y)
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(input_x, input_y, test_size=0.1)

'''
for i in range(10):

	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(input_x, input_y, test_size=0.1)

	linear = linear_model.LinearRegression()

	linear.fit(x_train,y_train)
	acc = linear.score(x_test,y_test)
	print("Accuracy",acc)
	if acc > best:
		best = acc
		with open("heart.pickle","wb") as f:
			pickle.dump(linear,f)

print("Best one",best)
'''	
pickle_in = open("heart.pickle", "rb")
linear = pickle.load(pickle_in)
acc = linear.score(x_test, y_test)
print("Accuracy ", acc)
prediction_ = linear.predict(x_test)
print(y_test)
for i in range(10):
	print("Actual",y_test[i])
	print("Predicted chances of Disease",prediction_[i]*100)
	print("Predicted chances of  not having Disease",(1 - prediction_[i])*100)
	
	print("***********")
