
#%%

import pandas as pd
import numpy as np

import os

os.getcwd()
# %%
input_dir = os.path.join(os.getcwd(),'data')


# %%

OMX = pd.read_csv(os.path.join(input_dir, 'omxs30.csv'))
NIFTY = pd.read_csv(os.path.join(input_dir, 'NSEI2.csv'))

#%%

print(OMX.head(5))
print(NIFTY.head(5))

print('Nifty shape: ' + str(NIFTY.shape))

print('OMX shape: ' + str(OMX.shape))

# %%

for data in [OMX, NIFTY]:
    data.drop['Open', 'High', 'Low', 'Close', 'Volume']


# %%

# %%
