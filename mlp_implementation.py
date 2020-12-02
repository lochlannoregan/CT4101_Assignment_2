import numpy as np
import misc


def init(X_train, y_train, hidden_layers_parameters):

    hidden_layers = populate_hidden_layers(hidden_layers_parameters)

    print(hidden_layers)

    # Depends on preprocessing for one hot encoding
    output_layer = generate_output_layer(y_train)

    input_data_row(X_train, y_train)


def populate_hidden_layers(hidden_layers_parameters):
    """Creates the hidden layers and initialises the weights to random values in the range (0,1)

    Args:
        hidden_layers_parameters (list): The lists contains integers representing the number of neurons for a
                                         corresponding layer

    Returns:
        list: a list of hidden layers in lists with random weights
    """

    hidden_layers = []
    number_of_layers = len(hidden_layers_parameters)
    for i in range(number_of_layers):
        hidden_layer = []
        weights = []
        generator = np.random.default_rng()
        for j in range(hidden_layers_parameters[i]):
            weights.append(generator.random())
        hidden_layer.append(weights)
        hidden_layers.append(hidden_layer)
    return hidden_layers


def generate_output_layer(y_train):
    pass


def input_data_row(input_var, output_var):
    pass


def print_network(network):
    # Visually represent it for debugging?
    pass

