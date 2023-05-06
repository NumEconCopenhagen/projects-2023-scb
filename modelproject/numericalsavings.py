
#%%
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
        
        t = 0
        while t < par.simT:

            if par.production_function == 'cobb-douglas':
                Y[t] = (K[t]**par.alpha)*(H[t]**par.phi)*(A[t]*L[t])**(1-par.alpha-par.phi)
            else:
                Y[t] = np.nan

            A[t+1] = A[t]*(1+par.g)     
            L[t+1] = L[t]*(1+par.n) 

            H[t+1] = Y[t]*sH + (1-par.delta)*H[t]
            K[t+1] = Y[t]*sK + (1-par.delta)*H[t]
            
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
    
#%%
#    def cons_t(self, sK=0.07, sH=0.012):
#             consumption = (1-sK-sH)*self.find_steady_state(sK=sK, sH=sH, tol=1e-6)
#             return consumption
#
#    def find_opt_s(self):
#
#        opt = SimpleNamespace()
#
#        s_K = np.linspace(1e-8, 1, 10)
#        s_H = np.linspace(1e-8, 1, 10)
#
#        x = np.linspace(0,1,10)
#        s_K, s_H = np.meshgrid(x,x) # all combinations
#    
#        s_K = s_K.ravel() # vector
#        s_H = s_H.ravel()
#
#    
#        cons = self.cons_t(sk=s_K, sH=s_H)
#        # c. set to minus infinity if constraint is broken
#        I = (s_K+s_H > 1) # | is "or"
#        cons[I] = -np.inf
#    
#        # d. find maximizing argument
#        j = np.argmax(x)
#        
#        opt.s_H = s_H[j]
#        opt.s_K = s_K[j]
#       
#        # b. calculate utility
#
#        return opt
    



#%%
#
#    def prod_yss(self, x):
#        par = self.par
#
#        denominator = (par.n+par.g+par.delta+(par.n*par.g))
#        powersk = ((par.alpha)/(1-par.alpha-par.phi))
#        powersh = ((par.phi)/(1-par.alpha-par.phi))
#
#        sol = self.sol
#
#
#        sol.y = par.A*((x[0]/denominator)**powersk)*((x[1]/denominator)**powersh)
#        y = sol.y 
#        return y
#    
#    #%%
#    def cons_ss(self, x):
#            par = self.par
#            sol =  self.sol
#            sol.cons_ss = self.prod_yss(x)*(1-x[0]-x[1])
#            cons_ss = sol.cons_ss
#
#            return cons_ss
#        
#
#    def sol_golden_cons_ss(self, x, analytical=True):
#        
#        par = self.par
#
#        if analytical == True:
#            
#            cons = ({'type': 'ineq', 'fun': lambda x:  1- x[0] - x[1]})
#            bnds = ((1e-8,1), (1e-8, 1))
#
#            x0 = [1e-5, 1e-5]
#
#            self.prod_yss(x)
#
#            min_cons = lambda x: -1*self.cons_ss(x)
#            gg = optimize.minimize(min_cons, x0=x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-8)
#            return gg
#        else: 
#            print("Ikke analytisk")