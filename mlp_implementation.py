import numpy as np
import misc


def init(X_train, y_train, hidden_layers_parameters):

    hidden_layers = populate_hidden_layers(hidden_layers_parameters)

    print(hidden_layers)

    output_layer = populate_output_layer(y_train)

    print(output_layer)

    input_data_row(X_train, y_train)


def populate_hidden_layers(hidden_layers_parameters):
    """Creates the hidden layers and populates the weights the weights accordingly

    Args:
        hidden_layers_parameters (list): The lists contains integers representing the number of neurons for a
                                         corresponding layer

    Returns:
        list: a list representing the hidden layers containing list(s) with a list of random weights assigned
    """
    hidden_layers = []
    number_of_layers = len(hidden_layers_parameters)

    for i in range(number_of_layers):
        hidden_layers.append(populate_random_weights(hidden_layers_parameters[i]))

    return hidden_layers


def populate_output_layer(y_train):
    """Calculates the number of classes from the output of the training data and populates weights accordingly

    Args:
        y_train (numpy list): The lists contains the actual output classifications for all training instances

    Returns:
        list: a list representing the output layer containing a list with random weights assigned
    """
    classification_types = []
    for value in y_train:
        if value not in classification_types:
            classification_types.append(value)
    output_layer = populate_random_weights(len(classification_types))
    return output_layer


def populate_random_weights(number_of_weights):
    """Populates weights in a list as random values in the range (0,1)

    Args:
        number_of_weights (int): This int specifies how many weights to populate

    Returns:
        list: a list representing the layer containing a list with random weights assigned
    """
    layer = []
    weights = []
    generator = np.random.default_rng()
    for j in range(number_of_weights):
        weights.append(generator.random())
    layer.append(weights)
    return layer


def input_data_row(input_var, output_var):
    pass


def print_network(network):
    # Visually represent it for debugging?
    pass

