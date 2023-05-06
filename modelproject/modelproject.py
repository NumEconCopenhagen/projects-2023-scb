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
        par.alpha = 1/3
        par.phi = 1/3

        par.n = 0.01    
        par.g = 0.02
        par.delta = 0.05
        
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1
        par.Y_init = 1

        par.simT = 200

    
    def find_steady_state(self, sK=0.12, sH=0.07, tol=1e-6):

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
        
        t = 0
        while t < par.simT:

            if par.production_function == 'cobb-douglas':
                Y[t] = (K[t]**par.alpha)*(H[t]**par.phi)*(A[t]*L[t])**(1-par.alpha-par.phi)
            else:
                Y[t] = np.nan

            A[t+1] = A[t]*(1+par.g)     
            L[t+1] = L[t]*(1+par.n) 

            H[t+1] = Y[t]*sH + (1-par.delta)*H[t]
            K[t+1] = Y[t]*sK + (1-par.delta)*K[t]
            
            y_tilde[t] = Y[t]/(A[t]*L[t])
            k_tilde[t] = K[t]/(A[t]*L[t])
            h_tilde[t] = H[t]/(A[t]*L[t])

            if (t>1) and (k_tilde[t]-k_tilde[t-1] < tol) and (h_tilde[t]-h_tilde[t-1] < tol):
                print("we are breaking")
                break

            t += 1
        print(t)
        # print(y_tilde)
        return y_tilde[:t], k_tilde[:t], h_tilde[:t]
            