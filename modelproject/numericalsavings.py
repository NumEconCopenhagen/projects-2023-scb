from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace
from scipy import optimize

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
        y_t = np.empty(par.simT)

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
            y_t[t] = Y[t]/L[t]

            if (t>1) and (k_tilde[t]-k_tilde[t-1] < tol) and (h_tilde[t]-h_tilde[t-1] < tol):
                print("we are breaking")
                break

            t += 1
        print(t)
        # print(y_tilde)
        return y_tilde[:t], k_tilde[:t], h_tilde[:t], y_t[:t]
 
    def cons_t(self, sK=0.12, sH=0.07):
            a = self.find_steady_state(sK=sK, sH=sH)
            b = a[3]
            b = (1-sK-sH)*b
            return b
    
    def negative_cons(self, x):
        sim_result = self.cons_t(sK=x[0], sH=x[1])
        ct = -1 * sim_result[-1]
        return ct

    def find_opt_s(self, discrete=True):

        if discrete == True:
            s_K = np.linspace(1e-8, 1, 200)
            s_H = np.linspace(1e-8, 1, 200)

            sk_res = []
            sh_res = []
            cons_res = []            
            for i in s_K:
                for x in s_H:
                    if i + x >= 1:
                        pass
                    else:
                        sk_res += [i]
                        sh_res += [x]
                        cons = self.cons_t(sK=i, sH=x);
                        cons_res += [cons[-1]]

            optimal_cons_t = np.max(cons_res)
            index = cons_res.index(optimal_cons_t)
            sk_opt = sk_res[index]
            sh_opt = sk_res[index]

            print(f"Optimal savings rates are sK = {sk_opt} and sH = {sh_opt}", optimal_cons_t)
            return sk_opt, sh_opt
        else: 
            print("Optimal savings rate continoues solution")
            x = [0, 0]
            
            x0 = [0.2, 0.2]
            cons = ({'type': 'ineq', 'fun': lambda x:  1- x[0] - x[1]})
            bnds = ((1e-3,1), (1e-3, 1))
            solcont = optimize.minimize(self.negative_cons, x0=x0, constraints=cons, bounds=bnds, method='Nelder-Mead')

            return solcont
        
        #x = np.linspace(0,1,10)
        #s_K, s_H = np.meshgrid(x,x) # all combinations
    #
        #s_K = s_K.ravel() # vector
        #s_H = s_H.ravel()
#
    #
        #cons = self.cons_t(sK=s_K, sH=s_H)
#
        #cons = cons[-1]
        ## c. set to minus infinity if constraint is broken
        #I = (s_K+s_H >= 1) # | is "or"
#
        #cons[I] = -np.inf
    #
        ## d. find maximizing argument
        #j = np.argmax(x)
        #
        #opt.s_H = s_H[j]
        #opt.s_K = s_K[j]
       #
#
        #return opt
    



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