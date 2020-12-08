from misc import forward_propagation
from misc import sigmoid_derivative
import random


def init(X_train, y_train, hidden_layers_parameters, X_test, y_test):

    # Each sub list of network represents a layer in the network
    # The layer structure is layer[number_of_connections, connection_weights[], activations[], bias]
    network = []

    learning_rate = .3

    number_of_epochs = 500

    populate_input_layer(network, X_train)
    populate_hidden_layers(network, hidden_layers_parameters)
    populate_output_layer(network, y_train)

    for epochs in range(number_of_epochs):
        for index, row in X_train.iterrows():
            forward_propagation(network, row)
            calculate_error(network, index, y_train)
            update_network(network, learning_rate, row)

    number_of_training_examples = 0
    number_correct = 0

    for index, row in X_test.iterrows():
        number_of_training_examples += 1
        forward_propagation(network, row)
        predicted = model_prediction(network, y_test)
        actual = y_test.loc[index].idxmax()
        if predicted == actual:
            correct_boolean = True
            number_correct += 1
        else:
            correct_boolean = False
        print("Expected = " + predicted + " Actual = " + actual + " Are same: " + str(correct_boolean))
    print("Accuracy: " + str(number_correct/number_of_training_examples * 100))


def update_network(network, learning_rate, row):

    # print("hi")

    for neurons in range(network[1][0]):
        for connections in range(network[0][0]):
            row_value = row[connections]
            network[1][1][neurons][connections] = network[1][1][neurons][connections] + (learning_rate * network[1][4][neurons] * row[connections])
        network[1][3][neurons] = network[1][3][neurons] + (learning_rate * network[-1][4][neurons])


    # Adjusting weights for output layer
    # Outer neurons loop probably can get rid of may cause issues for more of these guys
    for neurons in range(network[1][0]):
        for connections in range(len(network[-1][1])):
            # Might have a problem here accessing connections for more than 1 neuron in hidden layer?
            # Might have problems here with no += ?
            network[-1][1][connections][0] = network[-1][1][connections][0] + (learning_rate * network[-1][4][connections] * network[1][2][neurons])
        # Update bias
            network[-1][3][connections] = network[-1][3][connections] + (learning_rate * network[-1][4][connections])


def calculate_error(network, index, y_train):
    # Should possibly be softmax or does that replace the activation function on the final layer?

    """Compares the predicted classification by the model to the actual classification and returns the error

    Args:
        network                 (list): This series of embedded lists represents the network
        y_train                 (pandas DataFrame): The lists contain the actual output classifications for all training
                                                 instances

    Returns:
        error                   (float): The summed difference of predicted and actual
    """
    number_output_neurons = network[-1][0]
    output_layer_errors = []
    for neurons in range(number_output_neurons):
        model_predicted_value = network[-1][2][neurons]
        actual_value = y_train.loc[index][neurons]
        error = (actual_value - model_predicted_value) * sigmoid_derivative(model_predicted_value)
        output_layer_errors.append(error)
    network[-1][4] = output_layer_errors

    # number_of_hidden_layers = len(network[1:-1])
    # for hidden_layer in range(1, number_of_hidden_layers + 1):
    # Code to iterate hidden layers
    for neurons in range(network[1][0]):
        error = 0.0
        # Having to calculate error in layer by neuron but doesn't seem to count for output layer - that's just the
        # difference between the output and predicted values so having to reach forward
        for connections in range(len(network[-1][1])):
            error += (network[-1][1][connections][0] * network[-1][4][connections])
        # See hardcoded number of neurons [0] in hidden layer at end
        network[1][4].append(error * sigmoid_derivative(network[1][2][0]))


def populate_input_layer(network, X_train):
    """Calculates the number of features and sets it in the input layer in the network

    Args:
        network                  (list): This series of embedded lists represents the network
        X_train                  (pandas DataFrame): The lists contain the actual input training data
    """
    features = []
    for value in X_train:
        if value not in features:
            features.append(value)
    network.append([len(features)])


def model_prediction(network, y_test):
    max_activation = max(network[-1][2])
    index_of_max_activation = network[-1][2].index(max_activation)
    prediction = y_test.columns[index_of_max_activation]
    return prediction


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
        y_train                  (pandas DataFrame): The lists contain the actual output classifications for all training
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
    bias_weights = []
    for i in range(number_of_neurons):
        weights = []
        for j in range(number_of_connections):
            weights.append(random.random())
        connection_weights.append(weights)
        bias_weights.append(random.random())
    layer.append(connection_weights)
    neurons = []
    for i in range(number_of_neurons):
        neurons.append(0)
    layer.append(neurons)
    layer.append(bias_weights)
    # adding space for computed errors
    layer.append([])
    return layer

