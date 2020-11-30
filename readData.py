# Importing required packages
import pandas as pd
import numpy as np

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