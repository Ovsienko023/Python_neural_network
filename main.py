from typing import List

import numpy
import scipy.special
import csv
import matplotlib
import matplotlib.pyplot as plt

from until import png_in_bin


class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.wih = (numpy.random.normal(0.0, pow(self.hidden_nodes, - 0.5), (self.hidden_nodes, self.input_nodes)))
        self.who = (numpy.random.normal(0.0, pow(self.output_nodes, - 0.5), (self.output_nodes, self.hidden_nodes)))

        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid
        # self.who = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        pass


    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # calculate error
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                                   numpy.transpose(hidden_outputs))

        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                                   numpy.transpose(inputs))

        pass


    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
        final_outputs = numpy.array(targets_list, ndmin=2).T

        final_inputs = self.inverse_activation_function(final_outputs)
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_outputs = self.inverse_activation_function(hidden_outputs)

        inputs = numpy.dot(self.wih.T, hidden_inputs)
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

def main():
    #neural_network = NeuralNetwork(784, 200, 10, 0.1)
    neural_network = NeuralNetwork(784, 100, 10, 0.1)
    # print(neural_network.query([1.0, 0.5, -1.5]))

    train_lines: List[str]
    with open("Tren_csv\\mnist_train.csv") as f:
        train_lines = f.readlines()

    tested_lines: List[str]
    with open("Tren_csv\\mnist_test.csv") as t:
        tested_lines = t.readlines()

    net_train(neural_network, train_lines)
    efficiency = net_test(neural_network, tested_lines)
    print("Efficiency:", efficiency)
    # show_plt(train_lines)  # показать картинку 
    label = 0
    targets = numpy.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    print(targets)
    image_data = neural_network.backquery(targets)
    

def net_train(neural_network, train_lines):
    epochs = 1
    for epoch in range(epochs):
        for train_line in train_lines:
            line_values = train_line.split(',')
            inputs = (numpy.asfarray(line_values[1:]) / 255.0 * 0.99) + 0.01
            targets = numpy.zeros(10) + 0.01  # 10- outputs_nodes
            targets[int(line_values[0])] = 0.99
            neural_network.train(inputs, targets)
            #print_targets(all_values, neural_network)


def net_test(neural_network, tested_lines):
    scorecard = list()  # list of estimates
    for record in tested_lines:
        tested_values = record.split(',')
        original_value = int(tested_values[0])
        #print("Original value:", original_value)

        inputs = (numpy.asfarray(tested_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = neural_network.query(inputs)

        label = numpy.argmax(outputs)  # index of the biggest value
        #print("Network respond:", label)
        if (label == original_value):
            scorecard.append(1)
        else:
            scorecard.append(0)
    print(scorecard)
    scorecard_array = numpy.asfarray(scorecard)
    efficiency = scorecard_array.sum() / scorecard_array.size
    return efficiency


def print_targets(all_values, neural_network):
    scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # TODO ОДНА  цифра 5 значения от 0.01 до 0.99
    print(len(scaled_input))
    targets = numpy.zeros(neural_network.output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    print(targets)


def show_plt(data_file_10):
    all_values = data_file_10[0].split(",")
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    print(image_array[0][0])
    plt.imshow(image_array, cmap="Greys", interpolation='None')
    plt.show()


main()




# date_new_png = png_in_bin.png_in_str()

