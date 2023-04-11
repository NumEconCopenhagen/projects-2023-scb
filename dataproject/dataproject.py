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
    rename_dict = {}
    for i in df.columns:
        rename_dict[i] = i.lower()
        rename_dict[i] = rename_dict[i].replace(' ','_')
    df = df.rename(columns=rename_dict)
    return df

