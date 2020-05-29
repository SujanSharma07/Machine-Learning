import tensorflow as tf


class AccuracyCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc') > 0.998):
      print ("Reached 99.8% accuracy so cancelling training")
      self.model.stop_training = True
      

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images/255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

callback = AccuracyCallback()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=20, callbacks=[callback])

evaluate = model.evaluate(test_images, test_labels)

print ("The accuracy in test set is ", evaluate[1])
