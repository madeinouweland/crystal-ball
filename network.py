import numpy as np


class Network:
    def __init__(self, training_data):
        self.input_neurons = []
        self.output_neurons = []
        self.connector_groups = []

        for line in training_data:
            for k in line[0]:
                i = next((i for i in self.input_neurons if i.name == k), None)
                if i is None:
                    self.input_neurons.append(Neuron(k))
            output_name = line[1]
            if output_name is not None:
                o = next((o for o in self.output_neurons if o.name == output_name), None)
                if o is None:
                    self.output_neurons.append(Neuron(output_name))

        self.connector_groups = [ConnectorGroup(self.input_neurons, o) for o in self.output_neurons]

    def print_network(self):
        for x in self.connector_groups:
            print(f'          conn. group {x.output_neuron.name}, b{x.bias:+0.6f}, cst{x.cost:+0.6f}')
            for c in x.connectors:
                print(f'{c.input_neuron.name:8} ({c.input_neuron.value:+0.2f}) ---------- w{c.weight:+0.6f} ----------> {c.output_neuron.name} ({c.output_neuron.value:+0.10f})')

    # take input key/values, map them on the input neurons
    # and propagate all connection groups
    def propagate(self, input):
        for x in self.input_neurons:
            x.value = 0
        for key in input:
            n = next(i for i in self.input_neurons if i.name == key)
            n.value = input[key]
        for cg in self.connector_groups:
            cg.propagate()

    # takes 1 line of training data, propagates and backpropagates
    def train(self, data_line):
        self.propagate(data_line[0])
        for cg in self.connector_groups:
            if cg.output_neuron.name == data_line[1]:
                target_value = 1
            else:
                target_value = 0
            cg.back_propagate(target_value)


class Neuron:
    def __init__(self, name):
        self.name = name
        self.value = 0


class Connector:
    def __init__(self, input_neuron, output_neuron):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.weight = np.random.randn()


class ConnectorGroup:
    def __init__(self, input_neurons, output_neuron):
        self.output_neuron = output_neuron
        self.connectors = [Connector(i, output_neuron) for i in input_neurons]
        self.bias = np.random.randn()
        self.cost = 0
        self.total_cost = 0

    def propagate(self):
        self.z = np.dot([c.input_neuron.value for c in self.connectors], [c.weight for c in self.connectors]) + self.bias
        self.output_neuron.value = sigmoid(self.z)

    def back_propagate(self, target_value):
        learning_rate = 0.2
        self.cost = np.square(self.output_neuron.value - target_value)
        self.total_cost = self.total_cost + self.cost

        dcost_pred = 2 * (self.output_neuron.value - target_value)
        dpred_dz = sigmoid_prime(self.z)
        dz_db = 1
        dcost_dz = dcost_pred * dpred_dz
        for c in self.connectors:
            dcost_dw = dcost_dz * c.input_neuron.value
            c.weight = c.weight - learning_rate * dcost_dw
        dcost_db = dcost_dz * dz_db
        self.bias = self.bias - learning_rate * dcost_db


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
