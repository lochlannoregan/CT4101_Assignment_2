import random
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from cycler import cycler


# Lochlann & Jack
def init(X_train, y_train, hidden_layers_parameters, X_test, y_test, learning_rate, epochs, output_file):
    """This init function accepts the parameters and data for running the algorithm before executing the steps to train
    a model, test against the testing dataset and output the results.

    Args:
        X_train                  (pandas DataFrame): This contains the input training data
        y_train                  (pandas DataFrame): This contains the output classifications for all
                                                    training instances
        hidden_layers_parameters (list): List containing the number of neurons to generate in the hidden layer
        X_test                   (pandas DataFrame): This contains the input testing data
        y_test                   (pandas DataFrame): This contains the output classifications for all
                                                    testing instances
        learning_rate            (float): Decides how fast with each step the weights and biases are adjusted when
                                          trying to approach the minimum of the error function
        epochs                   (int): This specifies how many times as part of training the training dataset will
                                        be passed through the network, error calculated and backpropagated appropriately
    """
    # Each sub list of network represents a layer in the network
    # The layer structure is layer[number_of_connections, connection_weights[], activations[], bias, error]
    network = []

    populate_input_layer(network, X_train)
    populate_hidden_layers(network, hidden_layers_parameters)
    populate_output_layer(network, y_train)

    n_zero_errors = 0

    learning_accuracys = list()

    for epoch in range(epochs):
        sum_error = 0

        # if there are 10 epochs with 100% train accuracy in a row, finish training
        if n_zero_errors == 10:
            break

        for index, row in X_train.iterrows():
            outputs = forward_propagation(network, row)
            calculate_error(network, index, y_train)
            update_network(network, learning_rate, row)

            if np.argmax(outputs) == 0:
                output_style = "ale"
            elif np.argmax(outputs) == 1:
                output_style = "lager"
            else:
                output_style = "stout"

            actual = y_train.loc[index].idxmax()
            if actual != output_style:
                sum_error += 1

        n_zero_errors += n_zero_errors if sum_error == 0 else 0

        print("Epoch: " + str(epoch) + "\t Error: " + str(sum_error), end='\r')

        accuracy = (len(X_train)- sum_error) / (len(X_train)) * 100

        learning_accuracys.append(accuracy)

    graph_learning_curve(epochs, learning_accuracys)

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
        output_file.write("Expected = " + predicted + " Actual = " + actual + " Are same: " + str(correct_boolean) + "\n")
    output_file.write("Learning Rate: " + str(learning_rate) + "\n")
    accuracy = number_correct/number_of_training_examples * 100
    print("\t\tAccuracy: " + str(accuracy))
    output_file.write("Accuracy: " + str(accuracy) + "\n\n")

    return accuracy


# Lochlann O'Regan 17316753
def update_network(network, learning_rate, row):
    """Updates the network layers, weights and biases having calculated the error

    Args:
        network                  (list): This series of embedded lists represents the network
        learning_rate            (int):  This value affects how quickly the change in weight is applied
        row                      (pandas Series): This contains the input row on the current iteration of training, the
                                                  input values are required for adjusting the weight of connections in
                                                  the first hidden layer
    """
    # Adjusting weights for hidden layer
    for neurons in range(network[1][0]):
        for connections in range(network[0][0]):
            # Updating connection weight in hidden layer
            # Existing weight plus (learning rate * calculated error for neuron * value from output layer)
            network[1][1][neurons][connections] = network[1][1][neurons][connections] + (learning_rate * network[1][4][neurons] * row[connections])
        # Updating the bias weight adding to existing bias + (learning rate * calculated error for neuron)
        network[1][3][neurons] = network[1][3][neurons] + (learning_rate * network[1][4][neurons])

    # Adjusting weights for output layer
    for neurons in range(network[-1][0]):
        for connections in range(network[1][0]):
            # Existing weight + (learning rate * calculated neuron error * input activation from previous layer)
            network[-1][1][neurons][connections] = network[-1][1][neurons][connections] + (learning_rate * network[-1][4][neurons] * network[1][2][connections])
        # Update bias
        network[-1][3][neurons] = network[-1][3][neurons] + (learning_rate * network[-1][4][neurons])


# Lochlann & Jack
def calculate_error(network, index, y_train):
    """Calculates the error for the output layer and also backpropagates error to the hidden layer

    Args:
        network                  (list): This series of embedded lists represents the network
        index                    (int):  The index allows the actual classification for the training row to be retrieved
                                         from y_train
        y_train                  (pandas DataFrame): The lists contain the actual output classifications for all
                                                    training instances
    """
    # Calculate output layer errors
    number_output_neurons = network[-1][0]
    output_layer_errors = []
    for neurons in range(number_output_neurons):
        model_predicted_value = network[-1][2][neurons]
        actual_value = y_train.loc[index][neurons]
        error = (actual_value - model_predicted_value) * sigmoid_derivative(model_predicted_value)
        output_layer_errors.append(error)
    # Setting error for each of output layer neurons
    network[-1][4] = output_layer_errors

    # Backpropagation of error for hidden layer
    hidden_layer_errors = []
    for neurons in range(network[1][0]):
        error = 0.0
        # As the error is backpropagating from the output layer to the hidden layer have to access the weights from the
        # reverse direction hence looping over the connection weights from the output layer
        for connections in range(len(network[-1][1])):
            # Hard coded 0 here after connections - may be issue if more than 1 neuron in hidden layer
            connection_weight_output_to_hidden = network[-1][1][connections][0]
            error_in_output_layer_for_neuron = network[-1][4][connections]
            error += (connection_weight_output_to_hidden * error_in_output_layer_for_neuron)
        activation_output_current_neuron = network[1][2][neurons]
        # Setting error for each of hidden layer neurons
        hidden_layer_errors.append(error * sigmoid_derivative(activation_output_current_neuron))
    network[1][4] = hidden_layer_errors


# Lochlann O'Regan 17316753
def populate_input_layer(network, X_train):
    """Calculates the number of features and sets it in the input layer in the network

    Args:
        network                  (list): This series of embedded lists represents the network
        X_train                  (pandas DataFrame): This contains the actual input training data
    """
    features = []
    for value in X_train:
        if value not in features:
            features.append(value)
    network.append([len(features)])


# Lochlann O'Regan 17316753
def model_prediction(network, y_test):
    """Returns the prediction by the trained model having passed through a test data row

    Args:
        network                  (list): This series of embedded lists represents the network
        y_test                   (pandas DataFrame): The lists contain the actual output classifications for all
                                        testing instances

    """
    max_activation = max(network[-1][2])
    index_of_max_activation = network[-1][2].index(max_activation)
    prediction = y_test.columns[index_of_max_activation]
    return prediction


# Lochlann O'Regan 17316753
def populate_hidden_layers(network, hidden_layers_parameters):
    """Creates the hidden layers and calls populate_values to populate the weights accordingly

    Args:
        network                  (list): This series of embedded lists represents the network
        hidden_layers_parameters (list): The list contains integers representing the number of neurons for a
                                         corresponding layer
    """
    number_of_layers = len(hidden_layers_parameters)

    for i in range(number_of_layers):
        number_of_connections = network[0+i][0]
        network.append(populate_values(number_of_connections, hidden_layers_parameters[i]))


# Lochlann O'Regan 17316753
def populate_output_layer(network, y_train):
    """Calculates the number of classes from the output of the training data and calls populate_values

    Args:
        network                  (list): This series of embedded lists represents the network
        y_train                  (pandas DataFrame): The lists contain the actual output classifications for all
                                                    training instances
    """
    classification_types = set()
    for value in y_train:
        classification_types.add(value)
    network.append(populate_values(network[-1][0], len(classification_types)))


# Lochlann O'Regan 17316753
def populate_values(number_of_connections, number_of_neurons):
    """Populates number of connections, connection weights, activations and bias into the network

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


# Jack Lynch 17370591
def activation(inputs, weights, bias):
    """Summation of weights times inputs including bias to calculate activation of a neuron

    Args:
        inputs                   (pandas Series): This contains the input training data / activations depending on the
                                                  activation is calculated for the input layer or a hidden layer
        weights                  (list): The weights on connections for a neuron is provided in a list
        bias                     (float): The bias provided is the bias for a specific neuron

    Returns:
        list: a list representing the layer containing a list with values assigned
    """
    outputs = bias
    for i in range(len(weights)):
        outputs += (weights[i] * inputs[i])
    return outputs


# Jack Lynch 17370591
def sigmoid(x):
    """Passes the supplied value through the non-linear sigmoid function

    Args:
        x                       (float): The x value is the input to the sigmoid function

    Returns:
        _                       (float): The output of the sigmoid function is returned
    """
    return 1/(1+math.exp(-x))


# Jack Lynch 17370591
def forward_propagation(network, inputs):
    """Enumerates the network and sums the inputs, weights and biases before passing them through a non-linear
    activation function to calculate the activation of neurons having passed an input through the network

    Args:
        network                  (list): This series of embedded lists represents the network
        inputs                   (pandas Series / list): Depends on whether the inputs comes to the input layer or
                                                         hidden layer
    Returns:
        inputs                   (list): This contains the calculated activations for neurons within the network
    """
    for index, layer in enumerate(network):
        if index != 0:
            activations = []
            for index_second, neuron in enumerate(layer[1]):
                neuron_activation = sigmoid(activation(inputs, neuron, layer[3][index_second]))
                activations.append(neuron_activation)
            inputs = activations
            layer[2] = activations
    return inputs


# Jack Lynch 17370591
def sigmoid_derivative(x):
    """Derivative of the sigmoid function calculation
    Args:
        x                       (float): This value is passed through the sigmoid derivative

    Returns:
        _                       (float): This value is returned as the output of the calculation
    """
    return x * (1.0 - x)


# Jack Lynch 17370591
def graph_learning_curve(n_epochs, y):

    x = list(range(1, n_epochs+1))
    y = y
    
    plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'b', 'y','c', 'm', 'y', 'k']) )

    plt.plot(x, y, linewidth = 1)

    plt.xlim(1,n_epochs) 
    plt.ylim(1,100)

    plt.xlabel('Iterations')
    plt.ylabel('Learning Accuracy (%)')

    plt.title('Learning Curve  ' + str(datetime.datetime.now()))

    plt.savefig('./data/learning_curve.png')
