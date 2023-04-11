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
    for i in df.columns: # loop over column names:
        # remove lower case and spaces from column names
        rename_dict[i] = i.lower()
        rename_dict[i] = rename_dict[i].replace(' ','_')
    df = df.rename(columns=rename_dict)
    # Create variables for causal analysi
    # s
    df['daily_return'] = (df['adj_close']-df['adj_close'].shift(1))/df['adj_close'].shift(1)
    # df['mean_return'] = df['daily_return'].mean()
    # df['deameaned_return'] = df['daily_return'] - df['mean_return']

    return df

# def merge_clean(dataset):
#     """Calculate mean returns and demeaned returns for merged dataset"""
    
#     df = dataset
#     rename_dict = {}


#     df['mean_return_x'] =