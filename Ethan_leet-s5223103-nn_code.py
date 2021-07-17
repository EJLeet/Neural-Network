""" 
Ethan Leet
s5223103
2802 ICT
Assignment 2
Task 3 - Neural-Network
"""

import numpy as np
import pandas as pd
import sys
import gzip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from timeit import default_timer as timer


n_input = int(sys.argv[1])
n_hidden = int(sys.argv[2])
n_output = int(sys.argv[3])


def one_hot(digit):
    # one hot encoding function to turn all values 0 except for label digit
    label = np.zeros(10)
    label[int(digit)] = 1.0
    return label


# load data
print("Loading Data...")
x_train = np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")[1:, 1:]
x_train = x_train / 255.0  # normalise
y_train = [one_hot(y)
           for y in np.genfromtxt(gzip.open(sys.argv[4], "rb"), delimiter=",")[1:, 0]]

x_test = np.genfromtxt(gzip.open(sys.argv[5], "rb"), delimiter=",")[1:, 1:]
x_test = x_test / 255.0  # normalise
y_test = [one_hot(y)
          for y in np.genfromtxt(gzip.open(sys.argv[5], "rb"), delimiter=",")[1:, 0]]


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Sigmoid derivative function
def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)


# create NN class
class NeuralNetwork():
    def __init__(self, sizes, epochs = 30, learning_rate = 3.0, batch_size = 20, params = None):
        # params: user defined parameters for task 4
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.accuracy_per_epoch = []
        self.quadratic_cost_per_epoch = []
        self.cross_entropy_per_epoch = []
        self.weights, self.biases = self.initial_parameters(params)

    def initial_parameters(self, param):
        # Initialise prameters
        input_layer = self.sizes[0]
        hidden = self.sizes[1]
        output_layer = self.sizes[2]
        if param:
            # Task 4 optionals
            weight_1, weight_2, bias_1, bias_2 = param
            weights = {'Weight_1': weight_1, 'Weight_2': weight_2}
            biases = {'Bias_1': bias_1, 'Bias_2': bias_2}
        else:
            # # Initialise biases and weights
            weights = {
                'Weight_1': np.random.randn(hidden, input_layer) * np.sqrt(1. / hidden),
                'Weight_2': np.random.randn(output_layer, hidden) * np.sqrt(1. / output_layer),
            }
            biases = {
                'Bias_1': np.random.randn(hidden) * np.sqrt(1. / hidden),
                'Bias_2': np.random.randn(output_layer) * np.sqrt(1. / output_layer),
            }
        return weights, biases

    def forward_pass(self, x):
        # find activations through sigmoid
        weights = self.weights
        biases = self.biases
        weights['Activation_0'] = x
        weights['Raw_1'] = np.dot(weights["Weight_1"], weights['Activation_0']) + biases['Bias_1']
        weights['Activation_1'] = sigmoid(weights['Raw_1'])
        weights['Raw_2'] = np.dot(weights["Weight_2"], weights['Activation_1']) + biases['Bias_2']
        weights['Activation_2'] = sigmoid(weights['Raw_2'])
        # Return the activations of the output layer
        return weights['Activation_2']

    def backward_pass(self, y, output):
        # calculacte partial dereivatives
        weights = self.weights
        bias_prime = {}
        weight_prime = {}
        error = (output - y)
        n_output_nodes = self.sizes[2]
        n_hidden_nodes = self.sizes[1]
        n_input_nodes = self.sizes[0]
        raw_2 = error * sigmoid_prime(weights['Raw_2'])
        bias_prime['Bias_2'] = raw_2 * 1
        raw_2 = raw_2.reshape(n_output_nodes, 1)
        weight_prime['Weight_2'] = np.dot(raw_2, weights['Activation_1'].reshape(1, n_hidden_nodes))
        error = np.dot(weights['Weight_2'].T, raw_2)
        raw_1 = error * sigmoid_prime(weights['Raw_1']).reshape(n_hidden_nodes, 1)
        bias_prime['Bias_1'] = raw_1.reshape(n_hidden_nodes,) * 1
        weight_prime['Weight_1'] = np.dot(raw_1, weights['Activation_0'].reshape(1, n_input_nodes))
        return weight_prime, bias_prime

    def update_parameters(self, weight_prime_sum, bias_prime_sum):
        # Update network biases and weights using Stochastic Gradient Descent.
        for key, value in weight_prime_sum.items():
            self.weights[key] -= (self.learning_rate / self.batch_size) * value
        for key, value in bias_prime_sum.items():
            self.biases[key] -= (self.learning_rate / self.batch_size) * value

    def evaluate_test_data(self, x_test, y_test):
        # Find test data accuracy and costs for current network
        predictions = []
        partial_q_cost = np.zeros(self.sizes[-1])
        partial_ce_cost = np.zeros(self.sizes[-1])
        for x, y in zip(x_test, y_test):
            # Run x through network
            output = self.forward_pass(x)
            partial = np.argmax(output)
            y_digit = np.argmax(y)
            predictions.append((partial, y_digit))
            partial_q_cost += (output - y) ** 2
            partial_ce_cost += (y * np.log(output) + (1 - y) * np.log(1-output))
        # Prediction accuracy
        accuracy = sum(int(x == y)
                       for (x, y) in predictions) / len(predictions)
        self.accuracy_per_epoch.append(accuracy)
        # quadratic cost
        q_cost = np.sum(partial_q_cost) * (1 / (2 * len(x_test)))
        self.quadratic_cost_per_epoch.append(q_cost)
        # cross entropy cost
        ce_cost = np.sum(partial_ce_cost) * (1/len(x_test))
        self.cross_entropy_per_epoch.append(ce_cost)
        return accuracy

    def initialise_mini_batches(self, training_data):
        # set up mini batches for current networkis
        return [training_data[i:i + self.batch_size]
                for i in range(0, len(training_data), self.batch_size)]

    def stochastic_gradient_descent(self, x_train, y_train, x_test = None, y_test = None):
        # Split the training set into batches of size batch_size
        batches = self.initialise_mini_batches(list(zip(x_train, y_train)))
        epoch_accuracy = []
        for epoch in range(self.epochs):
            for batch in range(len(batches)):
                # Get all samples in the current batch for every epoch
                current_batch = batches[batch]
                # Summed errors of the weights and biases of the current batch
                sum_weight_prime = {'Weight_1': np.zeros((self.sizes[1], self.sizes[0])),
                               'Weight_2': np.zeros((self.sizes[2], self.sizes[1]))}
                sum_bias_prime = {'Bias_1': np.zeros(self.sizes[1]),
                               'Bias_2': np.zeros(self.sizes[2])}
                for x, y in current_batch:
                    # Feed input through network
                    output = self.forward_pass(x)
                    # Back propagate to get the partial derivatives
                    weight_prime, bias_prime = self.backward_pass(y, output)
                    # Sum the current weight errors
                    for key, val in weight_prime.items():
                        sum_weight_prime[key] += val
                    # Sum the current bias errors
                    for key, val in bias_prime.items():
                        sum_bias_prime[key] = np.add(sum_bias_prime[key], val)
                # Update the network params
                self.update_parameters(sum_weight_prime, sum_bias_prime)
            accuracy = self.evaluate_test_data(x_test, y_test)
            epoch_accuracy.append(accuracy * 100)
        return epoch_accuracy

    def predict(self, x_test):
        predictions = []
        for x in x_test:
            # Run x through network
            output = self.forward_pass(x)
            # Get highest probability
            pred = np.argmax(output)
            predictions.append(pred)
        return predictions


# Task 1
def predict_test_data(x_train, y_train, x_test, y_test, n_input, n_hidden, n_output):
    # Neural network with sizes=[784, 30, 10], epochs=30, batch_size=20 and l_rate=3.0
    network = NeuralNetwork(sizes=[n_input, n_hidden, n_output],
                            learning_rate = 3.0,
                            batch_size = 20,
                            epochs = 30)
    epoch_accuracy = network.stochastic_gradient_descent(
        x_train, y_train, x_test, y_test)
    return epoch_accuracy


def run(x_train, y_train, x_test, y_test, n_input, n_hidden, n_output):
    # Simple network for Task 1
    epoch_accruacy = predict_test_data(
        x_train, y_train, x_test, y_test, n_input, n_hidden, n_output)
    return epoch_accruacy

# Print data and plots pertaining to task 1
print("Task One: Running Neural Network of Size [784, 30, 10]")
epoch_accuracy = run(x_train, y_train, x_test, y_test,
                     n_input, n_hidden, n_output)
max_accuracy = max(epoch_accuracy)
print("Highest Accuracy Is: ", float(max_accuracy), "%")
# plotting accuracy vs epoch
plt.title('Neural Network [30, 20, 3.0] Accuracy vs Epoch')
plt.plot(epoch_accuracy, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('task1accuracy.png')


# task 2
def compare_learning_rates(x_train, y_train, x_test, y_test, n_input, n_hidden, n_output):
    # Run network with same settings a (1) except with learning rates [0.001, 0.1, 1.0, 10.0, 100.0]
    accuracies = []
    labels = []
    quadratic_costs = []
    cross_entropy_costs = []
    print(
        "\nPart 2: Training With Learning Rates [0.001, 0.1, 1.0, 10.0, 100.0]")
    for learning_rate in [0.001, 0.1, 1.0, 10.0, 100.0]:
        print("\nRunning Network With Learning Rate", learning_rate)
        network = NeuralNetwork(sizes=[n_input, n_hidden, n_output],
                                learning_rate=learning_rate,
                                batch_size = 20)
        epoch_accuracy = network.stochastic_gradient_descent(
            x_train, y_train, x_test, y_test)
        max_accuracy = max(epoch_accuracy)
        print("Highest Accuracy For Learning Rate", learning_rate,
              "Is:", "{:.2f}".format(float(max_accuracy)), "%")
        # Record costs
        accuracies.append(network.accuracy_per_epoch)
        # quadratic_costs.append(network.quadratic_cost_per_epoch)
        # cross_entropy_costs.append(network.cross_entropy_per_epoch)
        labels.append("learning_rate=" + str(learning_rate))
    return labels, accuracies


task_2_labels, task_2_accuracy = compare_learning_rates(
    x_train, y_train, x_test, y_test, n_input, n_hidden, n_output)
# Print data and plots pertaining to task 2
epoch = 30
x = range(1, epoch + 1)
# For Learning rates
for y, label in zip(task_2_accuracy, task_2_labels):
    plt.plot(x, y, label=label)
plt.title("Learning Rates")
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('task2accuracy.png')

# for y, label in zip(task_2_quadratic_costs, task_2_labels):
#     plt.plot(x, y, label=label)
# plt.title("Learning Rates - Quadratic Cost Function")
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.savefig('task2quadraticcost.png')

# # for y, label in zip(task_2_cross_entropy_costs, task_2_labels):
# #     plt.plot(x, y, label=label)
# # plt.title("Learning Rates - Cross Entropy Cost Function")
# # plt.legend()
# # plt.xlabel('Epoch')
# # plt.ylabel('Accuracy')
# # plt.savefig('task2crossentr.png')


# Task 3
def compare_batch_sizes(x_train, y_train, x_test, y_test, n_input, n_hidden, n_output):
    # Run network with same settings a (1) except with batch_sizes [1, 5, 10, 20, 100]
    accuracies = []
    labels = []
    quadratic_costs = []
    cross_entropy_costs = []
    print("\nPart 3: Training With Batch Sizes ")
    for batch_size in [1, 5, 10, 20, 100]:
        print("\nRunning Network With Batch Size", batch_size)
        start = timer()
        network = NeuralNetwork(sizes=[n_input, n_hidden, n_output], learning_rate=3.0,
                                batch_size=batch_size)
        epoch_accuracy = network.stochastic_gradient_descent(
            x_train, y_train, x_test, y_test)
        max_accuracy = max(epoch_accuracy)
        # Record costs
        accuracies.append(network.accuracy_per_epoch)
        # quadratic_costs.append(network.quadratic_cost_per_epoch)
        # cross_entropy_costs.append(network.cross_entropy_per_epoch)
        end = timer()
        labels.append("batch_size=" + str(batch_size))
        print("Highest Accuracy For Batch Size", batch_size,
              "Is:", "{:.2f}".format(float(max_accuracy)), "%")
        print("Batch Size", batch_size, "Took",
              "{:.2f}".format(end-start), "Seconds To Run")
    return labels, accuracies


task_3_labels, task_3_accuracy = compare_batch_sizes(
    x_train, y_train, x_test, y_test, n_input, n_hidden, n_output)

# Print data and plots pertaining to task 3
for y, label in zip(task_3_accuracy, task_3_labels):
    plt.plot(x, y, label=label)
plt.title("Batch Sizes")
plt.legend()
plt.xlabel('Mini-Batch Size')
plt.ylabel('Accuracy')
plt.savefig('task3accuracy.png')

# for y, label in zip(task_3_quadratic_costs, task_3_labels):
#     plt.plot(x, y, label=label)
# plt.title("Batch Sizes - Quadratic Cost Function")
# plt.legend()
# plt.xlabel('Mini-Batch Size')
# plt.ylabel('Accuracy')
# plt.savefig('task3quadraticcost.png')

# for y, label in zip(task_3_cross_entropy_costs, task_3_labels):
#     plt.plot(x, y, label=label)
# plt.title("Batch Sizes - Cross Entropy Cost Function")
# plt.legend()
# plt.xlabel('Mini-Batch Size')
# plt.ylabel('Accuracy')
# plt.savefig('task3crossentr.png')


# Task 4
def optional_hyper_params(x_train, y_train, x_test, y_test, n_input, n_hidden, n_output):
    # Run network with user defined hyper parameters
    learning_rate = 0.02
    batch_size = 64
    epochs = 50
    accuracies = []
    quadratic_costs = []
    cross_entropy_costs = []
    print("Task 4: Running Neural Network With Optional Hyper-Parameters")
    print("Learning Rate:", learning_rate)
    print('Batch Size: ', batch_size)
    print('Epochs: ', epochs)
    network = NeuralNetwork(sizes=[n_input, n_hidden, n_output], learning_rate=learning_rate,
                            batch_size=batch_size, epochs=epochs)
    epoch_accuracy = network.stochastic_gradient_descent(
        x_train, y_train, x_test, y_test)
    max_accuracy = max(epoch_accuracy)
    accuracies.append(network.accuracy_per_epoch)
    # quadratic_costs.append(network.quadratic_cost_per_epoch)
    # cross_entropy_costs.append(network.cross_entropy_per_epoch)
    print("Highest Accuracy For Optional Hyper-Parameters Is:",
          "{:.2f}".format(float(max_accuracy)), "%")

# Execute task 4
optional_hyper_params(x_train, y_train, x_test, y_test,
                      n_input, n_hidden, n_output)
