import pandas as pd
import os

def read_yahoo(input_dir, filename='file.csv'):
    """ Read csv file from Yahoo finance to df\n
    Args: 
        filename (filetype = csv), input_dir = directory of data
    Returns:
        Beautiful financial dataframe
    """
    df = pd.read_csv(os.path.join(input_dir, filename))
    df.columns=df.columns.lower().replace(' ','_')  # remove lower case and spaces from column names
    return df

