import os
import pandas as pd


def calculations(dataframe):
    for i in ['x','y']:
        dataframe['daily_return_{i}'] = ((dataframe['adj_close_{i}'])/(dataframe['adj_close{i}'].shift(1)))-1
    return dataframe