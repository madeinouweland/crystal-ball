from network import Network
import random


training_steps = 5000
training_data = [
    [{'cloudy': 0}, None],
    [{'cloudy': 1}, 'umbrella'],
]

network = Network(training_data)
network.print_network()

print('PREDICT UNTRAINED')
for x in training_data:
    network.propagate(x[0])
    network.print_network()

print('TRAIN')
for i in range(training_steps):
    network.train(random.choice(training_data))

print('PREDICT TRAINED')
for x in training_data:
    network.propagate(x[0])
    network.print_network()
