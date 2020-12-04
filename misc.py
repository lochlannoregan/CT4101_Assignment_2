import numpy as np


# for forward Propagate  
def activation(inputs,weights, bais):
    outputs = bais
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
                neuron_actitive = sigmoid(activation(inputs, neuron, layer[3]))
                activations.append(neuron_actitive)
            inputs = activations
            layer[2] = inputs
    return inputs




# for back Propagate 
def sigmoid_drivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# def backward_propagation(network, expected):
