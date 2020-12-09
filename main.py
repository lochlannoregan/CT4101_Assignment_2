# Importing required packages
from sys import argv
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlp_implementation

# sklearn neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler


def get_file_sep(file_name):
    file_extention = file_name.split('.')[-1]
    if file_extention.lower() == "csv":
        return ','
    else: 
        return '\t'


def load_data(file):

    file_name = file.split('/')[-1] if "/" in file else file

    sep = get_file_sep(file_name)

    # Loading Data
    data = pd.read_csv(file, sep=sep)

    if file_name == 'beer.txt':
        del data['beer_id']

    return data


def minipulate_data(data):
    # Partition Data into training and testing
    # length of beer random numbers, random uniform distribution 0-1
    # condition < 0.66, returns true for numbers less than .66
    # meaning a split of about 2/3 true, 1/3 false
    mask = np.random.rand(len(data)) < 0.66
    # beer[mask] equal to indexes that are true
    # beer[~mask] not equal to indexes that are true, i.e. indexes that are false
    data_train = data[mask]
    data_test = data[~mask]


    X_train = data_train.drop('style', axis=1)
    y_train = data_train['style']
    X_test = data_test.drop('style', axis=1)
    y_test = data_test['style']


    # # Not random testing version
    # train_set = beer.sample(frac=0.66, random_state=0)
    # test_set = beer.drop(train_set.index)
    #
    # # Separate the dataset as response variable and feature variables
    # X_train = train_set.drop('style', axis=1)
    # y_train = train_set['style']
    # X_test = test_set.drop('style', axis=1)
    # y_test = test_set['style']



    return X_train, y_train, X_test, y_test


def reference_algorithm(X_train, y_train, X_test, y_test, output_file):
    
    # Apply Standard scaling to get better results
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # sklearn neural network
    mlpc = MLPClassifier(learning_rate_init=0.07, activation='logistic', hidden_layer_sizes=5, max_iter=500)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)

    # print the models performance
    print("Accuracy: " + str(accuracy_score(y_test, pred_mlpc) * 100))
    output_file.write("Accuracy: " + str(accuracy_score(y_test, pred_mlpc) * 100) + "\n")
    output_file.write(str(classification_report(y_test, pred_mlpc)) + "\n")
    output_file.write(str(confusion_matrix(y_test, pred_mlpc)) + "\n\n")


def implementation_algorithm(X_train, y_train, X_test, y_test, output_file):
    normalized_X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

    normalized_X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    y_train_one_hot_encoding = pd.get_dummies(y_train, dtype=float)

    y_test_one_hot_encoding = pd.get_dummies(y_test, dtype=float)

    learning_rate = 0.07
    epochs = 5

    model_accuracy = mlp_implementation.init(normalized_X_train, y_train_one_hot_encoding, [5], normalized_X_test,
                                    y_test_one_hot_encoding, learning_rate, epochs, output_file)

    return model_accuracy



def graph_model_accuracy(accuracies):

    x = list(range(1, 11))
    y = accuracies

    fig, ax = plt.subplots()

    ax.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)

    ax.set_xlim(1,10) 
    ax.set_ylim(1,100)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy (%)')

    ax.set_title('Accuracies for each of the 10 runs: ' + str(datetime.datetime.now())) 
    
    ax.figure.savefig('./data/accuracies.png')

    print("Graph of Accurasscy over 10 runs: ./data/accuracies.png")




def run(data): 

    output_file_path = './data/data_output.txt'
    output_file = open(output_file_path, "w")
    output_file.write(str(datetime.datetime.now()) + "\n")
    
    accuracys = []

    for i in range(10):

        print("Iteration {}".format(i) )
        output_file.write("Iteration {} \n".format(i) )

        X_train, y_train, X_test, y_test = minipulate_data(data)

        print("implementation_algorithm")
        output_file.write("implementation_algorithm \n")
        model_accuracy = implementation_algorithm(X_train, y_train, X_test, y_test, output_file)
        accuracys.append(model_accuracy)

        print("reference_algorithm")
        output_file.write("reference_algorithm \n")
        reference_algorithm(X_train, y_train, X_test, y_test, output_file)
    
    print("Graph of Learning Curve: ./data/learning_curve.png")
    output_file.write("Graph of Learning Curve: ./data/learning_curve.png \n")

    graph_model_accuracy(accuracys)
    output_file.write("Graph of Accurasscy over 10 runs: ./data/accuracies.png \n")

    print("writien to file: {} \n".format(output_file_path))

    output_file.close()



def main():
   
    usage = "usage: main.py <file path>"

    input_file = ""

    if len(argv) == 2:
        input_file = argv[1]
    elif len(argv) > 2:
        print(usage)
        exit()
    else:
        input_file = './data/beer.txt'


    data = load_data(input_file)

    run(data)


if __name__ == "__main__":
    main()