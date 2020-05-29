import pandas as pd
import numpy as np
import sklearn
from sklearn import svm
import pickle


heart = pd.read_csv("heart.csv")



input_x = heart.drop(["target"], axis = 1)
ask_x = input_x.columns
drop = input_x.head(0)
input_x = np.array(input_x)
input_y = heart.drop(drop,axis =1)
input_y = np.array(input_y)
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(input_x, input_y, test_size=0.1)
'''
for i in range(10):

	x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(input_x, input_y, test_size=0.1)

	svm_model = svm.SVC(kernel = "linear")
	svm_model.fit(x_train,y_train)
	acc = svm_model.score(x_test, y_test)
	print("Acc :", acc)
	if acc > best:
		best = acc
		with open("heart_svm.pickle","wb") as f:
			pickle.dump(svm_model,f)
print("Best",best)	
'''		
pickle_in = open("heart_svm.pickle", "rb")
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

print(ask_x)
x = []
for i in ask_x:
	a=float(input("Enter the value of Patients "+ str(i)))
	x.append(a)
check = np.array(x).reshape(1, -1) 
prediction_ = linear.predict(check)
print("Predicted chances of Disease ",prediction_*100)
print("Predicted chances of  not having Disease ",(1 - prediction_)*100)
print("***********")

