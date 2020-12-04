import numpy as np


# for forward Propagate  
def activation(inputs, weights, bias):
    outputs = bias
    for i in range(len(weights)):
        outputs += weights[i] * inputs[i]
        # print(outputs)
    return outputs


def sigmoid(x):
    return 1/(1+np.exp(-x))


def forward_propagation(network, inputs):
    for index, layer in enumerate(network):
        # print(index)
        if index != 0:
            activations = []
            for neuron in layer[1]:
                neuron_activation = sigmoid(activation(inputs, neuron, layer[3]))
                activations.append(neuron_activation)
            inputs = activations
            layer[2] = inputs
    return inputs


# for back Propagate 
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))