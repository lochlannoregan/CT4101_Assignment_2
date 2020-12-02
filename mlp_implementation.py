import numpy as np
import misc
import HiddenLayer


def init(input_variables, number_hidden_layers, neurons_per_hidden_layer, output_variables):

    hidden_layers = []

    random_generator = np.random.default_rng()

    for i in range(number_hidden_layers):
        hidden_layer = {}
        for j in range(neurons_per_hidden_layer):
            hidden_layer.(j, random_generator.random())
        hidden_layers.append(hidden_layer)

    print(hidden_layers)

def input_data_row(variables, output):
    print("test")
