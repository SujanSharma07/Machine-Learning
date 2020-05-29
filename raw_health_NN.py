import numpy as np
import csv
import pandas as pd
import math


class NeuralNetwork():

    def __init__(self):
        # Seed the random number generator
        np.random.seed(1)

        # Set synaptic weights to a 3x1 matrix,
        # with values from -1 to 1 and mean 0
        self.synaptic_weights = 2 * np.random.random((13, 1)) - 1

    def sigmoid(self, x):
        """
        Takes in weighted sum of the inputs and normalizes
        them through between 0 and 1 through a sigmoid function

        """
        # print(max(0,x.all()))

        # return max(0,x.all())
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function used to
        calculate necessary weight adjustments
        """
        # return max(0,1)
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        We train the model through trial and error, adjusting the
        synaptic weights each time to get a better result
        """
        for iteration in range(training_iterations):
            # Pass training set through the neural network
            output = self.think(training_inputs)

            # Calculate the error rate
            error = training_outputs - output

            # Multiply error by input and gradient of the sigmoid function
            # Less confident weights are adjusted more through the nature of the function
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            # Adjust synaptic weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        """
        Pass inputs through the neural network to get output
        """

        inputs = inputs.astype(float)

        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output

    def ipdata(self, ih):

        x = np.zeros((304, 13), dtype=float)

        for i in range(210):
            for j in range(7):
                x[i][j] = ih.iloc[i][j]

        return x

    def opdata(self, ho):

        y = np.zeros((304, 1), dtype=float)

        for k in range(210):
            y[k] = ho.iloc[k][0]
        return y


if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # The training set, with 4 examples co0
    # nsisting of 3
    # input values and 1 output value

    # training_inputs = pd.read_csv("test.csv")
    # training_inputs.columns = ['a', 'b','c']

    df = pd.read_csv('heart.csv', skipinitialspace=True)
    training_inputs = df.drop('target',axis =1)[:304]
    test_input = df.drop('target',axis =1)[:30]

    training_inputs = neural_network.ipdata(training_inputs)
    # print(training_inputs)

    # print(training_inputs)
    # print("\n")
    training_outputs = df[['target']][:304]
    test_outputs = df[['target']][:30]
    training_outputs = neural_network.opdata(training_outputs)
    # print(training_outputs)
    # training_outputs = np.array([[0,1,1,0]]).T

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    print("Output data: ")
    count = 0
    for i in range(len(test_input)):a

        a = neural_network.think(np.array(test_input)[i])
        b = np.array(test_outputs)[i]
        if 1-a < 0.5 :
            print("Predicted is 1")
            print("Actual is",b)
        else:
            print("Predicted is 0")
            print("Actual is", b)
        if a == b:
            count +=1
    print("tested Accuracy of :",(count/30)*100)
