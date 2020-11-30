# Importing required packages
import pandas as pd
import numpy as np

# sklearn nerural network 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


def load_data():
    # Loading Data
    beer = pd.read_csv('./data/beer.txt', sep ='\t')
    del beer['beer_id']
    beer.info()

    # Partition Data into training and testing
    msk = np.random.rand(len(beer)) < 0.66
    beer_train = beer[msk]
    beer_test = beer[~msk]
    print('beer_train lentgh: ', len(beer_train))
    print('beer_test lentgh: ', len(beer_test))

    # Seperate the dataset as response variable and feature variables
    X_train = beer_train.drop('style', axis = 1)
    y_train = beer_train['style']
    X_test = beer_test.drop('style', axis = 1)
    y_test = beer_test['style']

    return X_train, y_train, X_test, y_test



# REFERENCE ALGORITHM 
def referance_algorithm(X_train, y_train, X_test, y_test):
    
    # Apply Standard scaling to get better results
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # sklearn nerural network 
    mlpc = MLPClassifier(hidden_layer_sizes=(9,12,9), max_iter=600)
    mlpc.fit(X_train, y_train)
    pred_mlpc = mlpc.predict(X_test)

    #print the models preformance
    print(classification_report(y_test, pred_mlpc))
    print(confusion_matrix(y_test, pred_mlpc))


# main function
def main():
    X_train, y_train, X_test, y_test = load_data()

    referance_algorithm(X_train, y_train, X_test, y_test)



if __name__ == "__main__":
    main()