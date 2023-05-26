import numpy as np
from scipy import optimize

# Define value function
def V(L, G, bl):
    """
    Calculates the value function.

    Args:
        L (float): Labor input.
        G (float): G value.
        bl: An object with parameters.

    Returns:
        float: Value function output.
    """
    return np.log(((bl.kappa + (1 - bl.tau) * bl.w * L) ** bl.alpha) * G ** (1 - bl.alpha)) - bl.nu * ((L * L) / 2)

def L_opt_analytical(bl, G):
    """
    Calculates the analytical solution for optimal labor input.

    Args:
        bl: An object with parameters.
        G (list): List of G values.

    Returns:
        list: List of optimal labor inputs.
    """
    L_opt = []
    for k in G:
        L_opt.append((-bl.kappa + np.sqrt(bl.kappa * bl.kappa + 4 * (bl.alpha / bl.nu) * bl.w_tilde * bl.w_tilde)) / (2 * bl.w_tilde))
    return L_opt

def L_opt(G, bl):
    """
    Calculates the numerical solution for optimal labor input.

    Args:
        G (list): List of G values.
        bl: An object with parameters.

    Returns:
        list: List of optimal labor inputs.
    """
    sol = []
    for k in G:
        obj = lambda L: -V(L, k, bl)
        x0 = 12
        sol.append(optimize.minimize(obj, x0, method='Nelder-Mead', bounds=((0, 24),)).x[0])
    return sol

# Define G
def G_func(L, bl): return bl.tau*bl.w*L

# Define analytical solution with definition for G substituted
def L_analytical_func(bl): 
    return (-bl.kappa + np.sqrt(bl.kappa * bl.kappa + 4 * (bl.alpha / bl.nu) * bl.w_tilde * bl.w_tilde)) / (2 * bl.w_tilde)

def iterate_over_tau(bl, do_print = False):

    # Create grid of taus and empty arrays
    tau_grid = np.linspace(1e-8,1,1000)
    L_vec = np.empty(1000)
    G_vec = np.empty(1000)
    V_vec = np.empty(1000)

    # Iterate L, G and V over tau
    for i, tau in enumerate(tau_grid):
        bl.tau = tau 
        bl.w_tilde = (1-bl.tau)*bl.w

        L_vec[i] = L_analytical_func(bl)
        
        G_vec[i] = G_func(L_vec[i], bl)
        
        V_vec[i] = V(L_vec[i], G_vec[i], bl) 
      
        if do_print == True: 
            print(f'for tau = {tau}, L = {L_vec[i]}, G = {G_vec[i]} and V = {V_vec[i]}')

    return tau_grid, L_vec, G_vec, V_vec 


#Define functions for private consumption and government spending
def calculate_C(bl, L):
    return bl.kappa + (1 - bl.tau) * bl.w * L

def calculate_G(bl, L, tau):
    # print(bl.tau,bl.w, L)
    return bl.tau*bl.w*L

# Define objective function
def objective_function(L, bl, G):
    C = calculate_C(bl, L)
    # G = calculate_G(bl,L)
    term1 = ((bl.alpha * C**((bl.sigma - 1) / bl.sigma) + (1 - bl.alpha) * G**((bl.sigma -1) / bl.sigma))**(bl.sigma / (bl.sigma-1)))**(1 - bl.rho) - 1
    term2 = bl.nu * L**(1 + bl.eps) / (1 + bl.eps)
    return -(term1 / (1 - bl.rho) - term2)

def solve_for_given_G(bl,G, tau):

    """
    Solves an optimization problem to find the value of 'L' for a given value of 'G' and 'tau'.

    Parameters:
        bl: An object or data structure representing some information.
        G: The given value of G.
        tau: The value of tau to be considered in the optimization problem. Default value is tau_opt.

    Returns:
        res: An object representing the result of the optimization.

    """
    
    # set tau and corresponding w_tilde
    bl.tau = tau
    bl.w = 1.0
    bl.w_tilde = (1-bl.tau)*bl.w

    # Define the bounds for L
    L_bounds = (1e-8, 24.0)

    # Set the initial guess for L
    x0 = 12

    # Define the optimization problem
    res = optimize.minimize(objective_function, x0, method = 'Nelder-Mead', args=(bl,G), bounds=([L_bounds]))    
   
    return res

def find_opt_G(bl,tau, do_print = False):

    """
        Finds optimal values of 'L' and calculates the difference between a range of 'G' values and tau*w*L.

        Parameters:
            bl: An object or data structure representing some information.
            tau: The value of tau to be considered in the optimization problem. Default value is tau_opt.
            do_print: A flag indicating whether to print additional information during the iteration. Default value is False.

        Returns:
            Gs: An array of 'G' values used in the iteration.
            Ls: An array of corresponding optimized values of 'L' for each 'G' value.
            diffs: An array of differences between 'G' and tau*w*L for each 'G' value.
            utility: The value of the objective function for the last iteration.

    """

    if do_print is True:
        print(bl)
        print(tau)

    # initialize empty numpy arrays
    Gs = np.linspace(0e-16,20,1000)
    Ls = np.empty(1000)
    diffs = np.empty(1000)
    utility = np.empty(1000)
    
    # iterate over grid og G
    for i, g in enumerate(Gs):
        res = solve_for_given_G(bl,g,tau)
        Ls[i] = res.x
        diffs[i] = g - tau*bl.w*Ls[i]
        
        utility[i] = -res.fun

        if do_print is True: print(f'for G = {g} --> L = {Ls[i]} and diff = {diffs[i]}')

    # reset parameter space    
    bl.tau = 0.3
    bl.w_tilde = (1-bl.tau)*bl.w

    return Gs, Ls, diffs, utility, tau