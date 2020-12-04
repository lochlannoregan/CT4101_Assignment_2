import numpy as np


# for forward Propagate  
def activation(inputs,weights):
    outputs = weights[-1]
    for i in range(len(weights)-1):
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
                neuron_actitive = sigmoid(activation(inputs, neuron))
                activations.append(neuron_actitive)
            inputs = activations
            layer[2] = inputs
    return inputs




# for back Propagate 
def sigmoid_drivative(x):
    return sigmoid(x) * (1 - sigmoid(x))