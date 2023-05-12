from scipy import optimize
import numpy as np
import time
from types import SimpleNamespace
import matplotlib
from matplotlib import pyplot as plt 
import ipywidgets as widgets
from ipywidgets import Output, SelectionSlider
from IPython.display import display, clear_output


class Solow():

    def __init__(self, do_ = True):
        """ define the model """

        if do_: ('initialising the model')

        self.par = SimpleNamespace()

        if do_: ('calling .setup()')
        self.setup()


    def setup(self):

        par = self.par

        par.production_function = 'cobb-douglas'
        par.alpha = 1/3
        par.phi = 1/3

        par.n = 0.01    
        par.g = 0.02
        par.delta = 0.05
        par.sH = 0.07
        par.sK = 0.12
        
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1

        par.simT = 200

    
    def find_steady_state(self, sK=0.12, sH=0.07, tol=1e-6, do_print=False):
        sim_out = self.sim_out = SimpleNamespace()
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
        
        steady_state_periods = []

        for i,j in zip([A,K,H,L,Y], [par.A_init, par.K_init,par.H_init,par.L_init]):
            i[0] = j
        
        t = 0
        while t < par.simT - 1:

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

            if (t>1) and (abs(k_tilde[t]-k_tilde[t-1]) < tol) and (abs(h_tilde[t]-h_tilde[t-1]) < tol):
                steady_state_periods += [t]
                if (do_print == True) and (t == steady_state_periods[0]):
                    print(f"Steady state reached in period {t}") 
            t += 1
        # (t)
        # (y_tilde)
        sim_out.y_tilde = y_tilde [:t-1]
        sim_out.k_tilde = k_tilde[:t-1]
        sim_out.h_tilde = h_tilde[:t-1]
        sim_out.y_t = y_t[:t-1]
        sim_out.A = A[:t-1]
        sim_out.K = K[:t-1]
        sim_out.H = H[:t-1]
        sim_out.L = L[:t-1]
        sim_out.steadystate_t = steady_state_periods[0]

        sim_out.t = np.linspace(0, len(y_tilde[:t-1]), 1)

        sim_out.sK = sK
        sim_out.sH = sH 

        return sim_out
    
    def anal_steady_state(self):
        anal_sol = self.sim_out = SimpleNamespace()
        par = self.par 

        u = par.n + par.g + par.delta + par.g*par.delta

        anal_sol.k_tilde = (par.sK/u)**((1-par.phi)/(1-par.phi-par.alpha))*(par.sH/u)**(par.phi/(1-par.phi-par.alpha))
        anal_sol.h_tilde = (par.sH/u)**((1-par.alpha)/(1-par.phi-par.alpha))*(par.sK/u)**(par.alpha/(1-par.phi-par.alpha))
        anal_sol.y_tilde = anal_sol.k_tilde**par.alpha*anal_sol.h_tilde**par.phi

        return anal_sol 

    def cons_t(self, sK=0.12, sH=0.07):
            
            """returns consumption from """
            sim_out= self.find_steady_state(sK=sK, sH=sH)
            y_t = sim_out.y_t
            consumption_t_vector = (1-sK-sH)*y_t
            return consumption_t_vector
    
    def negative_cons(self, x):
        sim_result = self.cons_t(sK=x[0], sH=x[1])
        ct = -1 * sim_result[-1]
        return ct

    def find_opt_s(self, discrete=True, discrete_sqrt_iter=50):
        """
        Returns: optimal savings rate for human and non human capital and consumption in period T
            
        Args: 
        discrete: bool, if True returns discrete solution from a grid search.
                        if False: Returns solution using scipy optimize. 
        discrete_sqrt_iter: square root of iterations in grid search for discrete solution.

        """
        self.sol_save = SimpleNamespace() #initialize simple namespace for solution
        sol_save = self.sol_save

        if discrete == True: 
            s_K = np.linspace(1e-8, 1, discrete_sqrt_iter) # Vector of possible capital saving rates
            s_H = np.linspace(1e-8, 1, discrete_sqrt_iter)  # Vector of possible human capital saving rates

            sk_res = [] #result vector
            sh_res = []
            cons_res = []            
            for i in s_K:
                for x in s_H:
                    if i + x >= 1: # constraint on saving rates
                        pass
                    else:
                        sk_res += [i]
                        sh_res += [x]
                        cons = self.cons_t(sK=i, sH=x);
                        cons_res += [cons[-1]]

            optimal_cons_t = np.max(cons_res) # find optimal solution
            index = cons_res.index(optimal_cons_t) # index of optimal solution
            sk_opt = sk_res[index] # save corresponding solution for s_K
            sh_opt = sk_res[index] # save corresponding solution for s_H

            (f"Optimal savings rates are sK = {sk_opt} and sH = {sh_opt}")
            sol_save.sK_opt = sk_opt # save optimal savings rate 
            sol_save.sH_opt = sh_opt # save optimal savings rate
            sol_save.cons_T_opt = optimal_cons_t #save optimal consumption in period T 
            sol_save.cons_T = cons_res
            sol_save.sK = sk_res
            sol_save.sH = sk_res
            return  sol_save
        
        else: 
            ("Optimal savings rate continoues solution")
            
            x0 = [0.2, 0.2] #initial values for optimisation
            cons = ({'type': 'ineq', 'fun': lambda x:  1- x[0] - x[1]}) #constraint (actually not used with 'Nelder-Mead' where a penalty function could have been implemented instead)
            bnds = ((1e-3,1), (1e-3, 1)) # bounds on the saving rates
            solcont = optimize.minimize(self.negative_cons, x0=x0, constraints=cons, bounds=bnds, method='Nelder-Mead') # call optimizer

            # pack solutions in namespace 
            sol_save.sK = solcont.x[0]
            sol_save.sH = solcont.x[1]
            sol_save.cons_t = solcont.fun

            return sol_save
        
    def plotbaseline_vs_new_sh(self, new_sH):
        """
        Returns: interactive plot comparing baseline with the post shock
        
        Args: New sH value not larger than 1 or smaller than 0.
        
        """
        par = self.par
        par.A_init = 1
        par.K_init = 1
        par.H_init = 1
        par.L_init = 1

        baseline_result = self.find_steady_state(sK=0.12, sH=0.07)
        
        ss_t = baseline_result.steadystate_t

        baseline_result.y_tilde = baseline_result.y_tilde[ss_t:]
        baseline_result.k_tilde = baseline_result.k_tilde[ss_t:]
        baseline_result.h_tilde = baseline_result.h_tilde[ss_t:]

        
        self.par.A_init = baseline_result.A[ss_t]
        self.par.K_init = baseline_result.K[ss_t]
        self.par.L_init = baseline_result.L[ss_t]
        self.par.H_init = baseline_result.H[ss_t]

        post_shock = self.find_steady_state(sK=0.12, sH=new_sH, do_print=False)

        post_shock_periods_index = int(self.par.simT)- 2 - len(baseline_result.k_tilde)
        
        post_shock.y_tilde  =  post_shock.y_tilde[:-post_shock_periods_index]
        post_shock.k_tilde  =  post_shock.k_tilde[:-post_shock_periods_index]
        post_shock.h_tilde  =  post_shock.h_tilde[:-post_shock_periods_index]
        plt.close('all')
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=120)
        ax.clear()
        ax.plot(baseline_result.y_tilde, label='y_tilde baseline')
        ax.plot(post_shock.y_tilde, label='y_tilde post shock')
        ax.plot(baseline_result.k_tilde, label='k_tilde baseline')
        ax.plot(post_shock.k_tilde, label='k_tilde post shock')
        ax.plot(baseline_result.h_tilde, label = 'h_tilde baseline')
        ax.plot(post_shock.h_tilde, label = 'h_tilde post shock')

        ax.legend(loc='upper right')
        plt.plot()


   
    def plotbaseline_vs_new_sh_intactive(self):
        out=widgets.interact(self.plotbaseline_vs_new_sh, new_sH=widgets.SelectionSlider(options=np.linspace(0,0.07,40), value=0))
        return display(out)
    
    def plot_convergence(self,H_init, K_init):
    
    felix = Solow()
    par = felix.par 
    par.simT = 1000
    # ii. change initial values (outside steady state)
   
    par.A_init = 1
    par.K_init = K_init
    par.H_init = H_init
    par.L_init = 1
    par.Y_init = 1

    # iii. extract simulation & unpack 
    sim_out = felix.find_steady_state() 
    k_t = sim_out.k_tilde
    h_t = sim_out.h_tilde

    # b. insert parameter values from simulation in nullclines  
    # i. define values 
    alpha_val = par.alpha
    delta_val = par.delta
    g_val = par.g
    n_val = par.n
    phi_val = par.phi
    sK_val = sim_out.sK
    sH_val = sim_out.sH

    # ii. find range of k_tilde
    k_tilde_vec = np.linspace(1e-10, max(k_t)+1, 100)

    # ii. insert in lamdified nullclines
    null_k_val = null_k_func(k_tilde_vec,alpha_val,delta_val,g_val, n_val, phi_val, sK_val)
    null_h_val = null_h_func(k_tilde_vec,alpha_val,delta_val,g_val, n_val, phi_val, sH_val)

        # c. plot results
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.plot(k_tilde_vec, null_k_val, label = r'$ \Delta \tilde{k}_t = 0$')
    ax.plot(k_tilde_vec, null_h_val, label = r'$ \Delta \tilde{h}_t = 0$')
    ax.plot(k_t, h_t, label='simulation', linestyle = "dotted", linewidth = 2)
    ax.set_xlabel(r'$\tilde{k}_t$',)
    ax.set_ylabel(r'$\tilde{h}_t$',)

    ax.legend(loc='upper left');
    
def convergence_interactive(self):
    out=widgets.interact(plot_convergence, H_init = widgets.SelectionSlider(options=np.linspace(0,5,40), value=5),
                         K_init = widgets.SelectionSlider(options=np.linspace(0,5,40), value=5))
    return display(out)