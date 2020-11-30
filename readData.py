#Importing required packages
import pandas as pd

#Loading Data
beer = pd.read_csv('./data/beer.txt', sep ='\t')
beer.info()