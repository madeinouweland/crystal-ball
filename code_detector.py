from network import Network
import random


training_steps = 50000
training_data = [
    [{'def': 2, 'print': 1, 'None': 2, 'for': 3}, 'python'],
    [{'def': 1, 'print': 2, 'None': 3, 'for': 4}, 'python'],
    [{'print': 1, 'for': 1}, 'python'],
    [{'def': 8}, 'python'],
    [{'function': 8, 'forEach': 1, ';': 6}, 'javascript'],
    [{'function': 1, 'forEach': 5, ';': 3, 'this': 7}, 'javascript'],
    [{'function': 3, 'forEach': 1, 'this': 1}, 'javascript'],
    [{'function': 2, 'forEach': 2, 'def': 2, ';': 7, 'this': 3}, 'javascript'],
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
    print(f'Predict {x}')
    network.propagate(x[0])
    network.print_network()
