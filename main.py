import numpy
import scipy.special
import csv
import matplotlib
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.wih = (numpy.random.normal(0.0, pow(self.hidden_nodes, - 0.5), (self.hidden_nodes, self.input_nodes)))
        self.who = (numpy.random.normal(0.0, pow(self.output_nodes, - 0.5), (self.output_nodes, self.hidden_nodes)))

        self.activation_function = lambda x: scipy.special.expit(x)  # сигмоида  
        # self.who = (numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        pass


    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # -- Вычисление ошибки
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                                    numpy.transpose(hidden_outputs))

        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                                    numpy.transpose(inputs))

        pass
    
    def query(self, inputs_list):  # опрос
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


n = NeuralNetwork(784, 100, 10, 0.3)
# print(n.query([1.0, 0.5, -1.5]))

data_file_10 = str()
with open("mnist_100.csv") as f:
    data_file_10 = f.readlines()

for i in data_file_10:
    all_values = i.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(10) + 0.01 # 10- outputs_nodes

    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

with open("mnist_10_test.csv") as t:
    test_data_list = t.readlines()

all_values = test_data_list[0].split(',')
print(all_values[0])


a = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
#print(a)
# ----------------------------------------------
all_values = data_file_10[0].split(",")
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
print(image_array[0][0])

plt.imshow(image_array, cmap="Greys", interpolation='None')
plt.show()
# -------------------------------------------------

# scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # ОДНА  цифра 5 значения от 0.01 до 0.99
# print(len(scaled_input))

# targets = numpy.zeros(output_nodes) + 0.01
# targets[int(all_values[0])] = 0.99
# print(targets)