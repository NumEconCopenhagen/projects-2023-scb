#%%
import pandas as pd
import numpy as np
import os
os.getcwd()


#%%
input_dir = os.path.join(os.getcwd(),'data')

#%%
nifty50 = pd.read_csv(os.path.join(input_dir,'OMXS30.csv'), sep=',')

# %%
nifty50.head()

#%%
nifty50['log_daily_return'] = np.log(nifty50['Adj Close']) - np.log(nifty50['Adj Close'].shift(1))

#%%
nifty50['daily_return'] = (nifty50['Adj Close']-nifty50['Adj Close'].shift(1))/nifty50['Adj Close'].shift(1)

# %%
nifty50.plot('Date', 'daily_return')

# %%
nifty50.head()

# %%
nifty50['dif'] = nifty50.daily_return - nifty50.log_daily_return

# %%
nifty50.plot('Date', 'dif') 


#%%
nifty50['dato'] = pd.to_datetime(nifty50['Date'])

nifty50.plot('dato', 'Volume')
# %%
nifty50_volume = nifty50.loc[nifty50.Volume>0, ['Volume', 'dato']]
# %%
nifty50_volume.plot('dato', 'Volume')

# %%
nifty50
# %%
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yfin


yfin.pdr_override()

spy = pdr.get_data_yahoo('SPY', start='2022-10-24', end='2022-12-23')


start = dt.datetime(2010, 1, 1)

end = dt.datetime.today()


actions = pdr.get_data_yahoo('GOOG', start, end)
