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
import ipywidgets as widgets


#%%
def plot_e(df, municipality): 
    I = df['municipality'] == municipality
    ax=df.loc[I,:].plot(x='year', y='empl', style='-o', legend=False)

widgets.interact(plot_e, 
    df = widgets.fixed(empl_long),
    municipality = widgets.Dropdown(description='Municipality', 
                                    options=empl_long.municipality.unique(), 
                                    value='Roskilde')
); 

#%%
from matplotlib import pyplot as plt


#%%
def say_my_name(name):
    """
    Print the current widget value in short sentence
    """
    print(f'My name is {name}')
     
widgets.interact(say_my_name, name=["Jim", "Emma", "Bond"]);


#%%
def plot_stock(df, stock='omx'):
    fig, ax = plt.subplots(nrow=1, ncols=1, figsize=(10,8))
    y = df[f'daily_return_{stock}']
    x = df['date']
    ax.plot(x, y)

widgets.interact
#%%

    
# %%
