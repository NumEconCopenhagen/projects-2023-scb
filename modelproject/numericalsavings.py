# %%
from scipy import optimize
import numpy as np
from types import SimpleNamespace

class Solow():

    def __init__(self, do_print = True):
        """ define the model """

        if do_print: print('initialising the model'):

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

#%%

    def setup(self):

        par = self.par

        par.alpha = 0.3
        par.phi = 0.3
        par.n = 0.01
        par.g = 0.01
        par.production_function = 'cobb-douglas'
        par.simT = 50
        par.K_init = 0.75
        par.H_init = 0.5
        par.L_init = 1
        par.A = 1

        par.s_K = 0.01
        par.s_H = 0.01


        sol = self.sol

#%%

    def prod_yss(self):
        par = self.par

        denominator = (par.n+par.g+par.delta+(par.n*par.g))
        powersk = ((par.alpha)/(1-par.alpha-par.phi))
        powersh = ((par.phi)/(1-par.alpha-par.phi))

        sol = self.sol

        sol.y = par.A*((par.s_K/denominator)**powersk)*((par.s_H/denominator)**powersh)

        return sol
    
    def cons_ss(x,self):
        
        par = self.par
        sol = self.sol

        x = [par.s_K, par.s_H]

        sol.cons_ss = sol.y*(1-x[0]-x[1])

        return sol.cons_ss
    
    def sol_golden_cons_ss(self):
        
        x0 = [1e-2, 1e-2]
        mincons = -1*self.cons_ss()
        gg = optimize.minimize(mincons, x0=x0, method='Nelder-Mead',)





# %%
