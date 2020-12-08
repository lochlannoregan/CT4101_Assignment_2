# Importing required packages
import pandas as pd
import numpy as np
import mlp_implementation

# sklearn neural network
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def load_data():
    # Loading Data
    beer = pd.read_csv('./data/beer.txt', sep='\t')
    del beer['beer_id']

    # Partition Data into training and testing
    # length of beer random numbers, random uniform distribution 0-1
    # condition < 0.66, returns true for numbers less than .66
    # meaning a split of about 2/3 true, 1/3 false
    msk = np.random.rand(len(beer)) < 0.66
    # beer[msk] equal to indexes that are true
    # beer[~msk] not equal to indexes that are true, i.e. indexes that are false
    beer_train = beer[msk]
    beer_test = beer[~msk]

    # Separate the dataset as response variable and feature variables
    X_train = beer_train.drop('style', axis=1)
    y_train = beer_train['style']
    X_test = beer_test.drop('style', axis=1)
    y_test = beer_test['style']

    return X_train, y_train, X_test, y_test



def reference_algorithm(X_train, y_train, X_test, y_test):
    
    # Apply Standard scaling to get better results
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # sklearn neural network
    mlpc = MLPClassifier(hidden_layer_sizes=(9, 12, 9), max_iter=600)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)

    # print the models performance
    print(classification_report(y_test, pred_mlpc))
    print(confusion_matrix(y_test, pred_mlpc))


def implementation_algorithm(X_train, y_train, X_test, y_test):
    normalized_X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min())

    normalized_X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min())

    y_train_one_hot_encoding = pd.get_dummies(y_train, dtype=float)

    y_test_one_hot_encoding = pd.get_dummies(y_test, dtype=float)

    model = mlp_implementation.init(normalized_X_train, y_train_one_hot_encoding, [1], normalized_X_test, y_test_one_hot_encoding)


def main():
    X_train, y_train, X_test, y_test = load_data()

    implementation_algorithm(X_train, y_train, X_test, y_test)

    # reference_algorithm(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()