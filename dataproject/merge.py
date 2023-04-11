
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import dataproject
import cleaner_func
os.getcwd()

# %%
input_dir = os.path.join(os.getcwd(),'data')

#%%

omx = dataproject.read_yahoo(input_dir, 'omxs30.csv')
nifty = dataproject.read_yahoo(input_dir, 'nsei2.csv')
dates = pd.read_pickle(os.path.join(input_dir, 'dates.pkl'))


# %%
for i in [omx, nifty]:
    i.drop(columns=['open', 'high', 'low','close', 'volume'], axis=1, inplace=True)
#%%
#First merge the two return data sets
merge_inner = pd.merge(omx, nifty, on='date',how='inner')


#%%
merge_inner =

#%%
merge_final = pd.merge(merge_inner,dates,on='date',how='left')
#%%
merge_final.plot('date',['daily_return_x', 'daily_return_y'], label = ['OMXS30', 'NIFTY50'],alpha=0.6)
# %%
