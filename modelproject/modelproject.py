from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace

class Solow():

    def __init__(self, do_print = True):
        """ define the model """

        if do_print: print('initialising the model'):

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):

        par = self.par

        par.alpha = 0.3
        par.phi = 0.3
        par.n = 0    
        par.g = 0
        par.production_function = 'cobb-douglas'
        par.simT = 50
        par.K_init = 0.75
        par.H_init = 0.5
        par.L_init = 1

    def find_opt_s(self):
        

    























# def solve_ss(alpha, c):
#     """ Example function. Solve for steady state k. 

#     Args:
#         c (float): costs
#         alpha (float): parameter

#     Returns:
#         result (RootResults): the solution represented as a RootResults object.

#     """ 
    
#     # a. Objective function, depends on k (endogenous) and c (exogenous).
#     f = lambda k: k**alpha - c
#     obj = lambda kss: kss - f(kss)

#     #. b. call root finder to find kss.
#     result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
#     return result