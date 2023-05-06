# %%
from scipy import optimize
import numpy as np
from types import SimpleNamespace

class Solow():

    def __init__(self, do_print = True):
        """ define the model """

        if do_print: print('initialising the model')
        self.sol = SimpleNamespace()
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):

        par = self.par
        sol = self.sol

        par.delta = 0.05
        par.alpha = 0.3
        par.phi = 0.2
        par.n = 0.01
        par.g = 0.01
        par.production_function = 'cobb-douglas'
        par.simT = 50
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1
        par.A = 1

        par.s_K = 0.01
        par.s_H = 0.01

        sol.y = 0
        sol.con_ss = 0

    def prod_yss(self, x):
        par = self.par

        denominator = (par.n+par.g+par.delta+(par.n*par.g))
        powersk = ((par.alpha)/(1-par.alpha-par.phi))
        powersh = ((par.phi)/(1-par.alpha-par.phi))

        sol = self.sol


        sol.y = par.A*((x[0]/denominator)**powersk)*((x[1]/denominator)**powersh)
        y = sol.y 
        return y
    

    #%%
    def cons_ss(self, x):
            par = self.par
            sol =  self.sol
            sol.cons_ss = self.prod_yss(x)*(1-x[0]-x[1])
            cons_ss = sol.cons_ss

            return cons_ss
        

    def sol_golden_cons_ss(self, x, analytical=True):
        
        par = self.par

        if analytical == True:
            
            cons = ({'type': 'ineq', 'fun': lambda x:  1- x[0] - x[1]})
            bnds = ((1e-8,1), (1e-8, 1))

            x0 = [1e-5, 1e-5]

            self.prod_yss(x)

            min_cons = lambda x: -1*self.cons_ss(x)
            gg = optimize.minimize(min_cons, x0=x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-8)
            return gg
        else: 
            print("Ikke analytisk")
 