#%% 
from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace


#%%
from numericalsavings import Solow

#%%
model = Solow()
# %%

model.find_steady_state()
# %%
