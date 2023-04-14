#%%
import pandas as pd
import numpy as np
import os
os.getcwd()

#%%
input_dir = os.path.join(os.getcwd(),'data')
nifty50 = pd.read_csv(os.path.join(input_dir,'NSEI2.csv'), sep=',')
nifty50.head()
rename_dict = {} #initiate dict

for i in nifty50.columns: #loop over names in columns
    # creates dict for renaming variables with no upper case letters nor any spaces
    rename_dict[i] = i.lower().replace(' ', '_') 
nifty50 = nifty50.rename(columns=rename_dict)

#%%
nifty50['log_daily_return'] = np.log(nifty50['adj_close']) - np.log(nifty50['adj_close'].shift(1))

#%%
nifty50['daily_return'] = (nifty50['adj_close']-nifty50['adj_close'].shift(1))/nifty50['adj_close'].shift(1)

# %%
nifty50.plot('date', 'daily_return')

# %%
nifty50.head()
nifty50['dif'] = nifty50.daily_return - nifty50.log_daily_return

# %%
nifty50.plot('date', 'dif')

#%%
nifty50.plot('date', 'volume')
# %%
nifty50_volume = nifty50.loc[nifty50.volume>0, ['volume', 'date']]
# %%
nifty50_volume.plot('date', 'volume')

# %%
easter_dates = pd.read_pickle(os.path.join(input_dir, 'dates.pkl'))
easter_dates.info()

#%%
nifty50['date'] = pd.to_datetime(nifty50['date'])
easter_dates['date'] = pd.to_datetime(easter_dates['date'])

#%%
nifty50_easter = pd.merge(nifty50, easter_dates, how='left', on='date')

#%%
nifty50_easter['easter_week'] = nifty50_easter['easter_week'].replace(np.nan, 0)
nifty50_easter['easter_week'].value_counts()

#%%
mean_return_easter_holiday = nifty50_easter.loc[nifty50_easter.easter_week==1, 'daily_return'].mean()
mean_return = nifty50_easter.loc[nifty50_easter.easter_week!=1, 'daily_return'].mean()
difference = mean_return_easter_holiday - mean_return

print(difference)
nifty50_easter.loc[nifty50_easter.easter_week==1].plot('date', 'daily_return', kind='kde')
nifty50_easter.loc[nifty50_easter.easter_week!=1].plot('date', 'daily_return', kind='kde')

#%%
import dataproject
import pandas as pd
input_dir = os.path.join(os.getcwd(),'data')
nifty50 = dataproject.read_yahoo(input_dir, filename='NSEI2.csv')

dataproject.read_yahoo()

# %% import using yfinance modulr
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin
import datetime as dt
yfin.pdr_override()

#%%
start = dt.datetime(2010, 1, 1)
end = dt.datetime.today()
nifty50 = pdr.get_data_yahoo('NFTY', start, end)

# %%
nifty50 = pdr.get_data_yahoo('NFTY', start, end)

# %%
nifty50

# %% Interactive plot varying years and varying and dataframe varying


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
merge_final.plot('date',['daily_return_omxs', 'daily_return_nifty'], label = ['OMXS30', 'NIFTY50'],alpha=0.6)
# %%

#%%
from matplotlib import pyplot as plt
import ipywidgets as widgets

    
# %%
def plot_stock_interactive(merge_final):
    def plot_stock_index(stock_index, start=min(merge_final.date), end=max(merge_final.date)):
        vertical_lines = merge_final.loc[merge_final.easter_week==1, 'date'].to_numpy()
        y_max = 0.075
        y_min = -0.11
        if start<= end and stock_index!='both':
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
            I = (merge_final.date>=start)&(merge_final.date<=end)
            x = merge_final.loc[I, 'date']
            y = merge_final.loc[I, f'daily_return_{stock_index}']
            ax.set_ylim(y_min, y_max)
            ax.plot(x, y, linewidth=1)
            ax.set_xlim(start, end)
            ax.set_xticks(x[::len(merge_final.loc[I,:])//10])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Daily returns for {stock_index}')
            ax.vlines(vertical_lines, y_min, y_max, color='#39ff14',
                       alpha=0.3,
                       linewidth=4, label="Easter period")
            ax.legend(loc="lower center")
        elif stock_index == 'both' and start <= end:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
            I = (merge_final.date>=start)&(merge_final.date<=end)
            for i in ['omxs', 'nifty']:
                x = merge_final.loc[I, 'date']
                y = merge_final.loc[I, f'daily_return_{i}']
                ax.set_ylim(y_min, y_max)
                ax.plot(x, y, linewidth=1)
            ax.set_xlim(start, end)
            ax.set_xticks(x[::len(merge_final.loc[I,:])//10])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.grid(True, alpha=0.3)
            ax.vlines(vertical_lines, y_min, y_max,
                       color='#39ff14',
                       alpha=0.3, linewidth=4, label="Easter period")
            ax.set_title(f"Daily returns for 'OMXS', 'NIFTY FIFTY'")
            ax.legend(loc="lower center")
        else: 
             print("Hello mate, please choose a valid range of dates")
        
    return widgets.interact(plot_stock_index, stock_index=['omxs', 'nifty', 'both'],
                              start = widgets.SelectionSlider(options=merge_final.date, step=0.04),
                                end = widgets.SelectionSlider(value=max(merge_final.date), options=merge_final.date), step=1);

# %%
plot_stock_interactive(merge_final)


# %%
