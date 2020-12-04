import numpy as np
from misc import  forward_propagation


def init(X_train, y_train, hidden_layers_parameters):

    # Each sub list of network represents a layer in the network
    # The layer structure is layer[number_of_connections, connection_weights[], activations[], bias]
    network = []

    populate_input_layer(network, X_train)
    populate_hidden_layers(network, hidden_layers_parameters)
    populate_output_layer(network, y_train)

    for index, row in X_train.iterrows():
        outputs = forward_propagation(network, row)
        print(outputs)

    print(network)
    

def populate_input_layer(network, X_train):
    """Calculates the number of features and sets it in the input layer in the network

    Args:
        network                  (list): This series of embedded lists represents the network
        X_train                  (numpy list): The lists contain the actual input training data
    """
    features = []
    for value in X_train:
        if value not in features:
            features.append(value)
    network.append([len(features)])


def populate_hidden_layers(network, hidden_layers_parameters):
    """Creates the hidden layers and calls populates the weights the weights accordingly

    Args:
        network                  (list): This series of embedded lists represents the network
        hidden_layers_parameters (list): The list contains integers representing the number of neurons for a
                                         corresponding layer
    """
    number_of_layers = len(hidden_layers_parameters)

    for i in range(number_of_layers):
        number_of_connections = network[0+i][0]
        network.append(populate_values(number_of_connections, hidden_layers_parameters[i]))


def populate_output_layer(network, y_train):
    """Calculates the number of classes from the output of the training data and calls populate_values

    Args:
        network                  (list): This series of embedded lists represents the networ
        y_train                  (numpy list): The lists contain the actual output classifications for all training
                                                instances
    """
    classification_types = []
    for value in y_train:
        if value not in classification_types:
            classification_types.append(value)
    network.append(populate_values(network[-1][0], len(classification_types)))


def populate_values(number_of_connections, number_of_neurons):
    """Populates number of connections, connection weights, activations and bias in a list

    Args:
        number_of_connections (int): This int specifies how many connections there are to each neuron
        number_of_neurons     (int): This int specifies how many neurons in this layer

    Returns:
        list: a list representing the layer containing a list with values assigned
    """
    layer = [number_of_neurons]
    connection_weights = []
    for i in range(number_of_neurons):
        weights = []
        for j in range(number_of_connections):
            weights.append(np.random.uniform(0, 1))
        connection_weights.append(weights)
    layer.append(connection_weights)
    neurons = []
    for i in range(number_of_neurons):
        neurons.append(0)
    bias_weight = np.random.uniform(0, 1)
    layer.append(neurons)
    layer.append(bias_weight)
    return layer

