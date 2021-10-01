# QM-gibbs
Code used to simulate and analyze the inverse-Wigner-Weyl transformation. Link to the [paper discussing the theory](https://arxiv.org/abs/2007.07264). 

The Classical mechanics side is solved in `CM_Solver_Wdiag_single.py`. It sets up the classical _W_ -matrix (described in Eq. 18 for the classical Gibbs distribution). Then, it diagonalizes that matrix to get the eigenvectors (the classical 'wave-functions') and eigenvalues (probabilities to be in those states). This computation is performed at certain temperature, $\beta$, and free-parameter, $\epsilon$. 

The Quantum mechanics side is solved in `QM_hoSolver_constField.py`. It sets up the Hamiltonian _H_-matrix for the cosine potential (but other potentials as necessary) and a secondary, time-dependent electric field. It then sets up the initial wave-function (one of the eigenstates of the time-independent Hamiltonian), maps that onto a Harmonic oscillator basis, and solves the dynamics in that basis. At each time-point, various observables are computed (e.g. `$<x^2>$`) to be compared to the classical case. 

The Jupyter notebooks perform different analysis, such as looking at the energy spectrum as a function of temperature, time-dependance of observables as a function of temperature, and comparing quantum and classical cases near the maxinally discrete point (where they match up best).  