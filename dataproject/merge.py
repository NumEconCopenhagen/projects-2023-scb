
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

import dataproject

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

#Then merge with the dummy for easter dates
merge_final = pd.merge(merge_inner,dates, on='date', how='left')
# %%
# %%
merge_final['mean_return_x'] = merge_final['daily_return_x'].mean()

merge_final['demeaned_return_x'] = merge_final['daily_return_x'] - merge_final['daily_return_x'].mean()

merge_final['mean_return_y'] = merge_final['daily_return_y'].mean()

merge_final['demeaned_return_y'] = merge_final['daily_return_y'] - merge_final['daily_return_y'].mean()
# %%
# %%

mean_x = merge_final['daily_return_x'].mean()

mean_easter_x = merge_final.loc[merge_final['easter_week']==1,'daily_return_x'].mean()

dif_x = mean_easter_x- mean_x 

mean_y = merge_final['daily_return_y'].mean()

mean_easter_y = merge_final.loc[merge_final['easter_week']==1,'daily_return_y'].mean()

dif_y = mean_easter_y - mean_y 

# %%
print('The difference in mean returns for OMXS30 is ' +str(dif_x))
print('The difference in mean returns for NIFTY50 is ' +str(dif_y))


# %%

fig, ax = plt.subplot(1,1,1)

merge_final.plot('date',['daily_return_x', 'daily_return_y'], label = ['OMXS30', 'NIFTY50'],alpha=0.6, ax=ax)
# %%
