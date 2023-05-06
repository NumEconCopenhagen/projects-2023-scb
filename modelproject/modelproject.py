from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace

class Solow():

    def __init__(self, do_print = True):
        """ define the model """

        if do_print: print('initialising the model')

        par = self.par = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()


    def setup(self):

        par = self.par

        par.production_function = 'cobb-douglas'
        par.alpha = 0.3
        par.phi = 0.3

        par.n = 0.01    
        par.g = 0.02
        par.delta = 0.05
        
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1
        par.Y_init = 1

        par.simT = 200



    
    def find_steady_state(self, sK=0.07, sH=0.12, tol=1e-6):

        par = self.par

        A = np.empty(par.simT)
        K = np.empty(par.simT)
        H = np.empty(par.simT)
        L = np.empty(par.simT)
        Y = np.empty(par.simT)
        y_tilde = np.empty(par.simT)
        k_tilde = np.empty(par.simT)
        h_tilde = np.empty(par.simT) 

        for i,j in zip([A,K,H,L,Y], [par.A_init, par.K_init,par.H_init,par.L_init,par.Y_init]):
            i[0] = j
        
        t = 1
        while t < par.simT:
       
            A[t] = A[t-1]*(1+par.g)     
            L[t] = L[t-1]*(1+par.n) 

            H[t] = Y[t-1]*sH + (1-par.delta)*H[t-1]
            K[t] = Y[t-1]*sK + (1-par.delta)*H[t-1]

            if par.production_function == 'cobb-douglas':
                Y[t] = (K[t]**par.alpha)*(H[t]**par.phi)*(A[t]*L[t])**(1-par.alpha-par.phi)
            else:
                Y[t] = np.nan
            
            y_tilde[t] = Y[t]/(A[t]*L[t])
            k_tilde[t] = K[t]/(A[t]*L[t])
            h_tilde[t] = H[t]/(A[t]*L[t])

            if (k_tilde[t]-k_tilde[t-1] < tol) and (h_tilde[t]-h_tilde[t-1] < tol):
                print("we are breaking")
                break

            t += 1
        print(t)
            




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