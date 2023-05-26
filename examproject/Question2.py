import numpy as np
from scipy import optimize

def profits(par, ell, kappa):
    return kappa*ell**(1-par.eta)-par.w*ell

# optimal ell based on analytical solution
def ell_opt_anal(par, kappa):
    """
    Calculates the optimal value of ell based on the analytical solution.

    Parameters:
    - par: Parameter namspace containing model parameters.
    - kappa: Array of kappa values.

    Returns:
    - ell_opt: Array of optimal ell values corresponding to each kappa value.
    """
    ell_opt = [None]*len(kappa)
    for i, k in enumerate(kappa):
        ell_opt[i] = ((1-par.eta)*k/par.w)**(1/par.eta)
    return ell_opt

# rule els based on rule for question 2.2 sames as analytical, but non list based inputs
def ell_opt_rulebased(par, kappa):
    """
    Calculates the optimal value of ell based on a rule-based formula.
    Parameters:
    - par: Parameter namespace containing model parameters.
    - kappa: kappa value.

    Returns:
    - ell_opt: optimal ell value corresponding to kappa value.
    """
    return ((1-par.eta)*kappa/par.w)**(1/par.eta)

# optimal ell 
def sol_ell(par, kappa):
    """ 
    Calculates optimal ell numerically for a list of given kappas
    Parameters:
    - par: simple namesspacem with model parameters 
    - kappa: array of kappas

    Returns
    - res: Array of results from optimization
    """
    res = [None] * len(kappa)
    #loop through kappas
    for i, k in enumerate(kappa):
        #define objective function 
        obj = lambda ell: -profits(par, ell, k)
        # call optimizer from scipy
        x0 = [0.1]
        res[i] = optimize.minimize(obj, x0, method='nelder-mead', bounds=((0, np.inf), ))
    return res

def kappa_series(par):
    """
    Calculates AR(1) series for given parameters
    Parameters:
    - par: simple namespace with model paramets
    Returns: 
    - kappa_ar1: AR(1) timeseries of kappa
    """

    eps = np.random.normal(loc=-0.5*par.sigma**2,scale=par.sigma, size=par.T) #create T shocks
    log_kappa_ar1 = np.zeros(par.T) #intitiate list for log kappas
    kappa_ar1 = np.zeros(par.T) # initiate list kappas 
    log_kappa_ar1[0] = par.rho*np.log(par.kappa_init)+eps[0] #log kappa0
    kappa_ar1[0] = np.exp(log_kappa_ar1[0]) #kappa 0
    for i in range(par.T):
        if i > 0: #calculate log kappa and kappa for period t
            log_kappa_ar1[i] = par.rho*np.log(kappa_ar1[i-1]) + eps[i]
            kappa_ar1[i] = np.exp(log_kappa_ar1[i])
        else: #let kappa0 be kappa0 
            kappa_ar1[i] = kappa_ar1[i]
            log_kappa_ar1[i] = log_kappa_ar1[i]
    return kappa_ar1

def h_func(par, shocks):
    """
    Calculates discounted profits for par.T periods, for given shocks
    Args:
    - par: SimpleNamespace with model parameters
    - shocks: AR(1) process of shocks

    Returns:
    - np.sum(profit_series): sum of discounted shocks.
    """

    profit_series = np.zeros(par.T)    
    ell_series = ell_opt_rulebased(par, shocks)

    for i in range(par.T):
        if i > 0 and np.abs(ell_series[i]-ell_series[i-1]) > par.Delta:
            profit_series[i] = (par.R**-i)*(profits(par, ell_series[i], shocks[i]) -par.iota)
        elif i > 0 and np.abs(ell_series[i]-ell_series[i-1]) <= par.Delta:
            ell_series[i] = ell_series[i-1] # in this case set ell_t to ell_t-1
            profit_series[i] = (par.R**-i)*profits(par, ell_series[i], shocks[i])
    return np.sum(profit_series)


def big_H(par):
    """
    Simulates profits par.K times
    Args:
    - par: SimpleNamespace with model parameters

    Returns:
    - np.mean(h_list): mean of simulated profits
    """
    h_list = np.zeros(par.K)
    for i in range(par.K):
        shocks = kappa_series(par)
        h_list[i] = h_func(par, shocks)
    return np.mean(h_list)

def obj(Delta, par):
    """
    Objective function used for optimizing choice Delta
    Returns:
    - negative of expected profits function
    """
    np.random.seed(1234)
    par.Delta = Delta
    return -1*big_H(par)
def sol_Delta(par):
    """
    Calculates optimal Delta
    Parameters:
    - par: SimpleNamespace with model parameters 
    """
    x0 = [0.01]
    res = optimize.minimize(obj, args=(par), x0=x0, method='nelder-mead')
    return res

def h_func_alt(par, shocks):
    """
    Calculates discounted profits for par.T periods, for given shocks
    Args:
    - par: SimpleNamespace with model parameters
    - shocks: AR(1) process of shocks

    Returns:
    - np.sum(profit_series): sum of discounted shocks.
    """

    profit_series = np.zeros(par.T)    
    ell_series = ell_opt_rulebased(par, shocks)

    k = 0
    for i in range(par.T):
        k+=1
        if k == par.gamma or i == 0: 
            profit_series[i] = (par.R**-i)*(profits(par, ell_series[i], shocks[i])-par.iota)
            k=0
        else:
            profit_series[i] = (par.R**-i)*(profits(par, ell_series[i-k], shocks[i])) 

    return np.sum(profit_series)

def big_H_alt(par):
    """
    Simulates profits par.K times
    Args:
    - par: SimpleNamespace with model parameters

    Returns:
    - np.mean(h_list): mean of simulated profits
    """
    h_list = np.zeros(par.K)
    for i in range(par.K):
        shocks = kappa_series(par)
        h_list[i] = h_func_alt(par, shocks)
    return np.mean(h_list)