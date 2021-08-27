# Graph Operations

import numpy as np

def adj(X,metric):
    """
    Computes the adjacency matrix given a set of molecules, X, and a particular distance metric.
    
    Parameters 
    ----------
    X : iterable 
        An iterable 
        
    metric : func
        A function that operates on two instances to compute their similiarty/distance
    """
    return np.array([[metric(x,y) for y in X] for x in X])

def lap(adjacency):
    """
    Computes the Laplacian from a given adjacency matrix 
    """
    adj = np.copy(adjacency)
    np.fill_diagonal(adj,0)
    adj *= -1
    np.fill_diagonal(adj,np.sum(adj,0)*-1)
    return adj 

def Fourier(arr,return_vals=False):
    """
    Computes and returns the fourier basis of a particular matrix. Assumes that the matrix is symmetric and uses np.eigh
    
    Parameters 
    ----------
    arr : np.array, (NxN)
        Assumed to be both square any symmetric in order to produce an orthogonal Fourier basis.
        
    return_vals : bool, default=False
        Whether or not to return the eigenvalues. Defaults to false as the point of this function is to compute the Fourier basis of a symmetric matrix.
        
    Returns
    -------
    q : np.array (NxN)
        Produces the eigenmatrix in which column i is the eigenvector corresponding to the ith eigenvector.
    """
    l, q = np.linalg.eigh(arr)
    if return_vals:
        return q, l
    else:
        return q
    
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
    
    return fourier_basis.T @ signal

def fourier_decomposition(X, K, S):
    r"""
    Computes the complete Fourier projection for a given set of representations (X), similarity measure, and signal defined over the graph.
    
    Parameters
    ----------
    X : np.array(NxM)
        An array containing N molecules with their associated M dimensional representation
        
    K : func 
        A similarity/kernel function that acts on X. It will return an X by X symmetric matrix of all possible pairwise similarities. Method must have 
        the same compute syntax as sklearn kernels or scipy's cdist module.
        
    S : np.array(N,)
        A signal in R^n that is going to projected onto the orthonormal eigenvases is K(X,X)
    
    Returns
    -------
    coeffs : np.array(N,)
        A numpy array of coefficients corresponding to the linear combination of eigenvectors required to reconstruct the signal.
    """
    adj = K(X,X)
    eigenvectors = Fourier(adj)
    return gft(eigenvectors,S)
