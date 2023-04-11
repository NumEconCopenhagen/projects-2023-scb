import os
import pandas as pd


def cleaner_func(dataframe):
    for i in ['x','y']:
        dataframe['mean_return_{i}']=dataframe['daily_return_{i}'].mean()
        dataframe['demeaned_return_{i}']=dataframe['daily_return_{i}']- dataframe['daily_return_{i}'].mean()
    return dataframe