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

try:
    plt.style.use('classic')
except:
    pass

## ================================================
## Static/Unperturbed system set-up

#Define parameters, potential and vector potential
m, omega , sigma, delta, Phi =  1., 1., 1., .2, 0.
eps = 0.1

def V(x):
    return omega*(1 - np.cos(x))

#Define range of x
x_max = np.pi
x_min = -np.pi
dx = 0.01
x_range_q = np.arange(x_min,x_max+dx,dx)
n_xq = len(x_range_q); 

#Construct Hamiltonian
H = np.zeros([n_xq,n_xq])
for x_i, x in enumerate(x_range_q):
    #Potential 
    H[x_i,x_i] += V(x)
    
    #Derivative
    H[x_i,x_i] -= -2. * (eps**2/(2.*m*dx**2))
    H[x_i,(x_i+1)%n_xq] -= 1. * (eps**2/(2.*m*dx**2))
    H[x_i,(x_i-1)%n_xq] -= 1. * (eps**2/(2.*m*dx**2))        

#Diagonalize H
[eig_vals_H,eig_vecs_H] = np.linalg.eigh(H)

## ========================================================================
## ========================================================================
## Dynamics set-up

## Define new parametesr for driving field
Ed = 0.01
omegaD = (eig_vals_H[1]-eig_vals_H[0])/eps
beta = 1/(eig_vals_H[1]-eig_vals_H[0])

def V_d(x, t):
    return Ed*x*np.cos(omegaD*t)


## (re)Define range of t
t_max = 100
t_min = 0
dt = 0.01
t_range_q = np.arange(t_min,t_max+dt,dt)
n_tq = len(t_range_q); 

print("Approx. period of RabiOsc.=", 2*np.pi * np.sqrt(2*m*(eig_vals_H[1]-eig_vals_H[0]) / Ed**2))
print("E1-E0=",eig_vals_H[1]-eig_vals_H[0])
print("beta=", beta, "; T=", 1/beta)


## Set up initial wave-function (making sure it's complex)
psi0 = np.zeros(n_xq, complex)
psi0 += 1*eig_vecs_H[:,10]

#Construct Hamiltonian
H_ho = np.zeros([n_xq,n_xq])
for x_i, x in enumerate(x_range_q):
    #Potential 
    H_ho[x_i,x_i] += omega * x**2 / 2
    
    #Derivative
    H_ho[x_i,x_i] -= -2. * (eps**2/(2.*m*dx**2))
    H_ho[x_i,(x_i+1)%n_xq] -= 1. * (eps**2/(2.*m*dx**2))
    H_ho[x_i,(x_i-1)%n_xq] -= 1. * (eps**2/(2.*m*dx**2))        

print("Check hermiticity (should be zero):", np.linalg.norm(H-np.conj(H.T)))

#Diagonalize H
[eig_vals_H_ho,eig_vecs_H_ho] = np.linalg.eigh(H_ho)

cutoff_ho = 400
omega0 = np.sqrt(omega / m)

#Construct x-operator in the HO basis
x_ho = np.zeros((cutoff_ho, cutoff_ho), "complex")
for x_i in range(cutoff_ho):
    if x_i+1 < cutoff_ho: 
        x_ho[x_i, x_i+1] = np.sqrt(eps/2/m/omega0) * np.sqrt(x_i+1)
        x_ho[(x_i+1)%cutoff_ho, x_i] = np.sqrt(eps/2/m/omega0) * np.sqrt(x_i+1)

def x_ho_power(power): 
    x_ho_return = x_ho.copy()
    for p in range(power-1): 
        x_ho_return = np.matmul(x_ho, x_ho_return)
    return x_ho_return

#Construct N-operator in the HO basis
n_ho = np.zeros((cutoff_ho, cutoff_ho))
for x_i in range(cutoff_ho):
    n_ho[x_i, x_i] = x_i+1

#Construct projections onto HO basis
c0 = np.zeros(cutoff_ho, 'complex')

for j in range(cutoff_ho): 
    c0[j] = np.sum(np.conj(eig_vecs_H_ho[:,j]) * psi0)

## ========================================================================
## ========================================================================

def schroEq_ho(t, c): 
    """
    Defines the differential equations for the TDSE w/ Driving.

    Arguments:
        t : time
        psi : vector of the wave-function at time t
    """
    c_Next = np.zeros(cutoff_ho, 'complex')
    c_Next += eps*omega0*(np.matmul(n_ho, c) - 0.5*c)
    c_Next -= omega*np.matmul(x_ho_power(4), c)/(4*3*2)
    c_Next += omega*np.matmul(x_ho_power(6), c)/(6*5*4*3*2)
    c_Next -= omega*np.matmul(x_ho_power(8), c)/(8*7*6*5*4*3*2)
    c_Next += omega*np.matmul(x_ho_power(10), c)/(10*9*8*7*6*5*4*3*2)
    c_Next += Ed * np.matmul(x_ho, c)
    return c_Next / (1j * eps)


## ================================================
## Solving Dynamics: n=0,...,100, and averaging each one's observable <x^2>

N_runs = 300
xVarAll = np.zeros((n_tq, N_runs))

for j in range(0, N_runs,1): 
    #Set up the wave-function
    psi0 = np.zeros(n_xq, complex)
    psi0 += 1*eig_vecs_H[:,j]
    
    #Construct projections onto HO basis
    c0 = np.zeros(cutoff_ho, 'complex')
    for k in range(cutoff_ho): 
        c0[k] = np.sum(np.conj(eig_vecs_H_ho[:,k]) * psi0)

    ## Call the ODE solver.
    print('integrating '+str(j), end='\r')
    cSol_ivp = solve_ivp(schroEq_ho, [t_range_q[0], t_range_q[-1]], c0, t_eval=t_range_q, method='RK45')
    cSol = np.transpose(cSol_ivp.y)
    
    ## Record the observable's evolution in time for a given j
    print('recording obervable '+str(j), end='\r')
    xVar_ho_j = np.zeros(n_tq)
    xVar_ho_fromWF_j = np.zeros(n_tq)
    for t_i, t in enumerate(t_range_q): 
        xVar_ho_j[t_i] += np.matmul(np.conj(cSol[t_i,:]), np.matmul(x_ho_power(2),cSol[t_i,:]))

    xVarAll[:,j] = xVar_ho_j

## Write the xVar Observable for each n to a text file
filename = "wave-functions/JJ_QM_xVar_ho_driveConst_eps0.1_trial100_c40.txt"
listToStr = ','.join([str(elem) for elem in [m, omega, eps, dx, dt, Ed, omegaD, beta, N_runs]])
np.savetxt(filename, xVarAll, delimiter=',', header=listToStr, comments='#')
