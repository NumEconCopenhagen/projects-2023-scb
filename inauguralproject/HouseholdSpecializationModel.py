
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3d figures

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """
        # KJT: 'sol' are solutions, 'par' are parameters.

        # a. create namespaces. 
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def __str__(self): # called when printed
        
        """" KJT: print baseline parameters at the moment """

        par = self.par 

        lines = '\n The baseline parameters in the model is: \n'
        lines += f'1. Preferences           = (rho, epsilon, omega, nu) = ({par.rho:.2f}, {par.epsilon:.2f}, {par.omega:.2f}, {par.nu:.3f}) \n'
        lines += f'2. Household production  = (alpha, sigma )           = ({par.alpha:.2f}, {par.sigma:.2f}) \n'
        lines += f'3. Wages                 = (wM,wF)                   = ({par.wM:.2f}, {par.wF:.2f}) \n'
               
        return lines

    def home_prod(self, HM, HF):
        """ calculate home production based on sigma value"""

        par = self.par

        if par.sigma == 0:
            H = np.fmin(HM,HF)
            return H
        
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
            return H

        else:
            potens = (par.sigma-1)/par.sigma 
            H = ((1-par.alpha)*HM**potens + par.alpha * HF**potens)**potens**-1
            return H 

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """
        
        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        H = self.home_prod(HM, HF)

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)

        # utility = np.fmax(Q,1e-8)**(1-par.rho)*(1-par.rho)**-1
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho) # RuntimeWarning: invalid value encountered in reciprocal

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            print('start solve_discrete()')
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
            print('end solve_discrete()')
        
        return opt

    def solve(self,do_print=False):
        """ solve model continously """
     
        opt_con = SimpleNamespace()

        # Now we use the library scipy to do the heavy lifting <3
        
        # a. objective function
        def value_of_choice(x):
            # note: x is an array, but calc_utility takes scalars
            # the array corresponts in order to LM, HM, LF, HF
            return -self.calc_utility(x[0], x[1], x[2], x[3])
        
        # b. constraints (violated if negative) and bounds. x is an array
        constraints = [{'type': 'ineq', 'fun': lambda x:  24-x[0]-x[1]},
                       {'type': 'ineq', 'fun': lambda x:  24-x[2]-x[3]}]
        bounds = ((1e-8,24),(1e-8,24),(1e-8,24),(1e-8,24))

        initial_guess = [1,1,1,1]

        # c. call solver, use SLSQP
        sol_case2 = optimize.minimize(
            value_of_choice, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints)
        
        # d. unpack solution
        opt_con.LM = sol_case2.x[0]
        opt_con.HM = sol_case2.x[1]
        opt_con.LF = sol_case2.x[2]
        opt_con.HF = sol_case2.x[3]

        if do_print:
            print('start solve()')
            for k,v in opt_con.__dict__.items():
                print(f'{k} = {v:6.4f}')
            print('end solve()')

        return opt_con

    def solve_wF_vec(self, discrete=False, do_print = False ):
        """ solve model for vector of female wages """

        par = self.par
        sol = self.sol

        for i, wF in enumerate(par.wF_vec):
            par.wF = wF

            if discrete:
                solution = self.solve_discrete(do_print=False)
            else:
                solution = self.solve(do_print=False)

            sol.LM_vec[i] = solution.LM
            sol.HM_vec[i] = solution.HM
            sol.LF_vec[i] = solution.LF
            sol.HF_vec[i] = solution.HF

        # b. reset wF value
        par.wF = 1 

        if do_print:
            print('start solve_wF_vec()')
            print(f'wf_vec = {par.wF_vec}')
            print(f'LM_vec = {sol.LM_vec}')
            print(f'HM_vec = {sol.HM_vec}')
            print(f'LF_vec = {sol.LF_vec}')
            print(f'HF_vec = {sol.HF_vec}')
            print('end solve_wF_vec()')

        print(f'in solve_wF_vec, HF_vec = {sol.HF_vec}')
        return sol 

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        print(f'in run_regression, HF_vec: {sol.HF_vec}')

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None, do_plot = False):
        """ estimate alpha and sigma """
        # its not right, since it really depends on initial guess

        est_sol = SimpleNamespace() 

        par = self.par
        sol = self.sol

        def min_function():
            return (sol.beta0-par.beta0_target)**2+(sol.beta1-par.beta1_target)**2
        
        # a. objective function
        def objective(x):
            alpha = x[0]
            sigma = x[1]

            par.alpha = alpha 
            par.sigma = sigma 

            self.solve_wF_vec()
            self.run_regression()
            print(f'min:function value {min_function()}')
            return min_function()
        
        # b. constraints (violated if negative) and bounds. x is an array
        # constraints = [{'type': 'ineq', 'fun': lambda x:  24-x[0]-x[1]},
        #                {'type': 'ineq', 'fun': lambda x:  24-x[2]-x[3]}]
        bounds = ((1e-8,1),(1e-8,10))

        initial_guess = [0.234,1]

        # change so we jump mote around
        options={'disp': True ,'eps' : 0.1, 'iter' : 25}

        # c. call solver, use SLSQP
        solution = optimize.minimize(objective, initial_guess,
                                    method='SLSQP', bounds=bounds, options=options)
        
        # d. unpack solution
        est_sol.alpha = solution.x[0]
        est_sol.sigma = solution.x[1]

        if do_plot:

            fig = plt.figure() # create the figure
            ax = fig.add_subplot(1,1,1,projection='3d') # create a 3d type axis 
            ax.plot_surface(x1_values,x2_values,u_values); # create surface plot in the axis
            # note: fig.add_subplot(a,b,c) creates the c'th subplot in a grid of a times b plots


        return est_sol


        

