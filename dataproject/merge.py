
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
rename_dict = {}
for i in merge_inner.columns:
        rename_dict[i] = i.lower()
        rename_dict[i] = rename_dict[i].replace('_x','_omxs')
        rename_dict[i] = rename_dict[i].replace('_y','_nifty')

merge_inner = merge_inner.rename(columns=rename_dict)

#%%
for i in ['omxs','nifty']:
      merge_inner[f'demeaned_return_{i}'] = merge_inner[f'daily_return_{i}'] - merge_inner[f'daily_return_{i}'].mean()
    #   merge_inner[f'mean_return_{i}'] = merge_inner[f'daily_return_{i}'].mean()


#%%
merge_inner['date'] = pd.to_datetime(merge_inner['date'])
dates['date'] = pd.to_datetime(dates['date'])
merge_final = pd.merge(merge_inner, dates,on='date',how='left')

#%%
merge_final.loc[merge_final['easter_week'].isna()==True, 'easter_week'] = 0 

mean_omxs = merge_final['daily_return_omxs'].mean()
mean_nifty = merge_final['daily_return_nifty'].mean()
# mean_omxs_easter = merge_final.loc[merge_final['easter_week' == 1]].mean()
# mean_nifty_easter = merge_final.loc['easter_week'==1,'daily_return_nifty'].mean()

#%%
pd.options.display.max_rows=100
merge_final.head(100)
#%%
merge_final.plot('date',['daily_return_x', 'daily_return_y'], label = ['OMXS30', 'NIFTY50'],alpha=0.6)
# %%
