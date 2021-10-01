import sys
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import scipy as sp
from scipy.integrate import solve_ivp
from random import choices

try:
    plt.style.use('ggplot')
except:
    pass

## ================================================
## Static/Unperturbed system set-up

#Define parameters, potential and vector potential
m, omega , sigma, delta, Phi =  1., 1., 1., .2, 0.
eps = 0.5

levelSpace = 0.46638204451990617 # eps = 0.5
beta = 0.01/ (levelSpace)

print("T=", 1/beta)

Ed = 0.01
omegaD = Ed / eps

def confin(x): 
    left = 20*(1+np.tanh(20*(x-np.pi)))
    right = 20*(1-np.tanh(20*(x+np.pi)))
    return left + right

def V(x):
    return omega*(1-np.cos(x)) + confin(x)

#Define range of x
x_max = np.pi
x_min = -np.pi
dx = 0.01
x_range_q = np.append(np.flip(-1*np.arange(0,x_max,dx)), np.arange(0,x_max,dx))
n_xq = len(x_range_q); 

#Define range of p
p_max = 15
p_min = -15
dp = 0.01
p_range_q = np.arange(p_min,p_max+dp,dp)
n_pq = len(p_range_q); 
print("n_pq=",n_pq )

## ================================================
## Define the function to compute density matrix

## Recall the classical probability distribution: P(x,p) == z[p, x]
## Now define function, $W$
def W(x_i, xi, z): 
    """
    Function that computes and returns the quasi-density matrix, W, for a given x in the P(x,p) and xi. 
    Arguments: 
        x_i : index of x-coordinate, which is also a x-coordinate in z[p_i, x_i]
        xi : value of xi-coordinate.
        z : classical probability distribution. 
    """
    w_x = np.sum( np.exp(-1j*p_range_q * xi/eps) * z[:, x_i] )
    return w_x

def W_sym(x1, x2, z): 
    """
    Function that computes and returns the quasi-density matrix, W, for a given x in the P(x,p) and xi. 
    Arguments: 
        x1 : x + xi/2
        x2 : x - xi/2
        z : classical probability distribution. 
    """
    x = (x1+x2)/2
    x_i = (np.abs(x_range_q - x)).argmin()
    w_x = np.sum( np.exp(-1j*p_range_q * (x1-x2)/eps) * z[:, x_i] )
    return w_x
    
    
## Set up system and functions 

def gibbs(x, p, beta): 
    P = np.exp(-beta * (p*p/(2*m) + omega*(1-np.cos(x))))
    return P

## ================================================
## Set up temperature grid and diagonalize 

def compStates(beta_list): 
    """
    Compute, plot, and return the weights from diagonalizing 
    the W-function (of the classical Gibbs distributions at inverse-temp beta)
    
    Arguments
        beta : inverse temperature. 
    """
    w_alph_list = []
    psi_alph_list = []
    
    for beta_i, beta in enumerate(beta_list): 
        printStr = "Working on " + str(beta_i) +" = "+str(beta)
        print(printStr)
        
        sigma_x = np.sqrt(beta*eps**2 / m)
        dx = 0.01

        #Define extended range of x1
        x1_max = np.pi
        x1_min = -np.pi
        x1_range_q = np.append(np.flip(-1*np.arange(0,x1_max+dx+5*sigma_x,dx)), np.arange(0,x1_max+dx+5*sigma_x,dx))
        # x1_range_q = np.arange(x1_min-5*sigma_x,x1_max+5*sigma_x+dx,dx)
        n_x1q = len(x1_range_q); 

        #Define extended range of x2
        x2_max = np.pi
        x2_min = -np.pi
        x2_range_q = np.append(np.flip(-1*np.arange(0,x2_max+dx+5*sigma_x,dx)), np.arange(0,x2_max+dx+5*sigma_x,dx))
        # x2_range_q = np.arange(x1_min-5*sigma_x,x1_max+5*sigma_x+dx,dx)
        n_x2q = len(x2_range_q);
        
        #Define range of x
        # Find the x-index where true x-range starts
        x_max = np.pi
        x_min = -np.pi
        x_range_q = np.append(np.flip(-1*np.arange(0,x_max,dx)), np.arange(0,x_max,dx))
        n_xq = len(x_range_q);
        found_x_beg = np.where(x1_range_q ==x_range_q[0])[0][0]
        found_x_end = -1*(found_x_beg)
        
        ## Compute the W-function
        W_mat = np.zeros((n_x1q, n_x2q), dtype=complex)
        Z_x = np.sum(np.exp(-beta*V(x_range_q)))
        for x1_i, x1 in enumerate(x1_range_q): 
            for x2_i, x2 in enumerate(x2_range_q):
                x_x = (x1+x2)/2
                xi_x = x1-x2
                W_mat[x1_i, x2_i] = np.exp(-beta*V(x_x))*np.exp(-m*xi_x**2/(2.*eps**2*beta))/Z_x

        ## Diagonalize the W-function
        w_alph, psi_alph = np.linalg.eigh(W_mat)
        w_alph_list += [w_alph]
        psi_alph_list += [psi_alph]
        
    return w_alph_list, psi_alph_list

## Call function to diagonalize 

params = sys.argv
beta_1 = float(params[1])

beta_list_wide = np.array([beta_1])  # narrow range of temperatures (despite the var name)

w_alph_wide_list, psi_alph_wide_list = compStates(beta_list_wide)

## Write the weights to a text file
filename = "diagW/JJ_CM_weights_nearMore" +str(round(beta_1, 6))+ ".txt"
listToStr = ','.join([str(elem) for elem in [m, omega, eps, beta]])
np.savetxt(filename, w_alph_wide_list[0], delimiter=',')

## Write the states to a text file
filename = "diagW/JJ_CM_states_nearMore" +str(round(beta_1, 6))+ ".txt"
listToStr = ','.join([str(elem) for elem in [m, omega, eps, beta]])
np.savetxt(filename, psi_alph_wide_list[0], delimiter=',', header=listToStr, comments='#')
