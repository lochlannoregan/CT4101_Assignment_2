import math

# for forward Propagate  
def activation(inputs, weights, bias):
    outputs = bias
    for i in range(len(weights)):
        outputs += (weights[i] * inputs[i])
        # print(outputs)
    return outputs


def sigmoid(x):
    return 1/(1+math.exp(-x))


def forward_propagation(network, inputs):
    for index, layer in enumerate(network):
        if index != 0:
            activations = []
            for index_second, neuron in enumerate(layer[1]):
                neuron_activation = sigmoid(activation(inputs, neuron, layer[3][index_second]))
                activations.append(neuron_activation)
            layer[2] = activations


# for back Propagate 
def sigmoid_derivative(x):
    return x * (1.0 - x)