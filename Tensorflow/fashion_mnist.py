import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
#normalization speed up the training speed
test_images = test_images / 255.0
model = keras.Sequential([
    #yo chai 28*28 ko data lai aautai flat list banauna ho
	#this is to convert 28*28 image to flat list of pixel values
    keras.layers.Flatten(input_shape=(28, 28)),

    #128 denote hidden layer neurons/ dense vanya totallly connected neurons to another layer
    keras.layers.Dense(128, activation='relu'),

    #yo chai op lai 10 possible output, softmax le chai k chai sab aako op ko value haru add garda 1 i.e 100% aauni banauxa
#this step converts all possible output values in probability values i.e sum of all value for each output will be 100%
#output is expressed in terms of probability
    #meaning op lai probability ko term ma express garna
    keras.layers.Dense(10, activation="softmax")
    ])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
'''
predictions = model(train_images[:1]).numpy()
loss_fn(train_labels[:1], predictions).numpy()
'''
model.compile(optimizer='adam',  loss=loss_fn, metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=8)
test_loss, test_acc = model.evaluate(test_images,test_labels, verbose = 2)
print("Tested Acc", test_acc)


#to save model..this will save training time while we just need to predict only
model.save("model_1.h5")
#to use saved model just use model.load("MODEL_NAME for our case model_1.h5")
#.h5 is the extension for saving model
#when we use pretrained model we do not need to fit our model

prediction = model.predict(test_images)
#predict garna chai ..test_images indicate the image to predict

#predicted_index = np.argmax(prediction[0])
#prediction matra garda huni theyo if aauta matra image vako vaye but
#hamro test_images ma tannai images xa so..1st ko matra check garam aaile

for i in range(5):
    predicted_index = np.argmax(prediction[i])
    #np.argmax le chai maximum probability vako prediction ko index return garxa
#np.argmax returns index of maximum value of a array
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual :"+ class_names[test_labels[i]])
    plt.title("Prediction :"+class_names[predicted_index] )
    plt.show()

