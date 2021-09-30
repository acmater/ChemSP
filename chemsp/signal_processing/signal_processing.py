"""
Provides a suite of basic graph operations to perform on matrix representations of chemical graphs.

Functions
========

fourier_basis : Returns the fourier basis (with optional eigenvalues) for a matrix representation of a graph
gft : Performs a graph Fourier transform on a signal using the eigenbasis (output of Fourier)
fourier_decomposition : Bundles multiple steps together, performs the GFT on a signal from the raw representations, X, similarity measure, K, and signal, S.
"""

import numpy as np
from chemsp.graphs import adjacency

def fourier_basis(gso,return_vals=False):
    """
    Computes and returns the fourier basis of a particular matrix. Assumes that the matrix is symmetric and uses np.eigh
    
    Parameters 
    ----------
    gso : np.array, (NxN)
       gso stands for Graph Shift Operator, of which this method should be able to work on any.
       Assumed to be both square any symmetric in order to produce an orthogonal Fourier basis.
        
    return_vals : bool, default=False
        Whether or not to return the eigenvalues. Defaults to false as the point of this function is to compute the Fourier basis of a symmetric matrix.
        
    Returns
    -------
    eigenvectors : np.array (NxN)
        Produces the eigenmatrix in which column i is the eigenvector corresponding to the ith eigenvector.
    """
    assert np.allclose(gso, gso.T), "The provided graph shift operator is not symmetric."
    assert gso.shape[0] == gso.shape[1], "The provided graph shift operator is not square."
    eigenvalues, eigenvectors = np.linalg.eigh(arr)
    if return_vals:
        return eigenvectors, eigenvalues
    else:
        return eigenvectors
    
def gft(fourier_basis, signal):
    """
    Performs the Graph Fourier Transform on a signal and returns the unsorted coefficient spectrum.
    
    Parameters
    ----------
    fourier_basis : np.array (NxN)
        The fourier basis of interest. Note that the columns must be the eigenvectors, not the rows. This is the default behaviour of numpy.eigh
        
    signal : np.array (N,)
        The signal to be projected onto the Fourier basis. Dimensions of the two must match
        
    Returns 
    -------
    coefficients : np.array (N,)
        The coefficient spectrum of the signal projected onto the Fourier basis.
    """
    assert fourier_basis.shape[0] == signal.shape[0], "The number of eigenvectors must equal the dimension of the signal."
    
    return fourier_basis.T @ signal # Compute the inner produce between the matrix of eigenvectors (Fourier basis) and the signal of interest.

def fourier_decomposition(X, K, S):
    """
    Computes the complete Fourier projection for a given set of representations (X), similarity measure (K), and a signal (S) defined over the graph.
    
    Parameters
    ----------
    X : np.array(NxM)
        An array containing N molecules with their associated M dimensional representation
        
    K : func 
        A similarity/kernel function that acts on X. It will return an N by N symmetric matrix of all possible pairwise similarities. Method must have 
        the same compute syntax as sklearn kernels or scipy's cdist module.
        
    S : np.array(N,)
        A N dimensional signal that is going to projected onto the orthonormal eigenbasis of K(X,X)
    
    Returns
    -------
    coeffs : np.array(N,)
        A numpy array of coefficients corresponding to the linear combination of eigenvectors required to reconstruct the signal.
    """
    adj = adjacency(X, K)
    eigenvectors = fourier_basis(adj)
    return gft(eigenvectors,S)
