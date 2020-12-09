# CT4101_Assignment_2

## Get started
- Usage:
    - main.py \<file path>
    - Or don't specify a file and will automatically use the provided beer.txt file in data/beer.txt
- Python version for development
    - Python 3.8
- The following libraries and versions through pip are required as of December 2020 to execute this code:
    - numpy 1.19.3
    - matplotlib 3.3.3
    - scikit-learn 0.23.2
    - pandas 1.1.5
    - Command:
        - ```pip install numpy==1.19.3 scikit-learn==0.23.2 pandas==1.1.5 matplotlib==3.3.3```

---

## Multi-Layer Perceptron Implementation
-  Pre-processing
    - weight initialisation
        - random values in range 0->1
    - feature scaling
        - min-max normalisation in range 0->1
    - One-hot encoding on target values
- Non-linear activation function
    - sigmoid
- Current hyper-parameters
    - epochs = 500
    - learning_rate = 0.07

---

## Reference Multi-Layer Perceptron Implementation
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
- Current hyper-parameters
    - max_iterations = 500
    - hidden_layer_sizes=(5)
        - 1 hidden layers with 5 neurons
    - learning_rate_init=0.07
    - activation='logistic' which is sigmoid function
