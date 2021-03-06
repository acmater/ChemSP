"""
Implements the necessary code to generate a variety of different graph shift operators given molecular representations and an adjacency metric.

Functions
=========

adj: Computes an adjacency matrix given a metric and a list of X values
lap: Computes a graph Laplacian matrix from an adjacency matrix.
"""
import numpy as np

def adjacency(X,metric):
    """
    Computes the adjacency matrix given a set of molecules, X, and a particular distance metric.
    
    Parameters 
    ----------
    X : iterable 
        An iterable 
        
    metric : func
        A function that operates on two instances to compute their similiarty/distance
    """
    if not isinstance(X,np.ndarray):
        try: X = np.array(X)
        except Exception as e:
            return e
    try:
        adj = metric(X,X)
        return adj
    except:
        return np.array([[metric(x,y) for y in X] for x in X])

def degree(adjacency):
    """
    Computes the degree matrix given an adjacency matrix.

    Parameters
    ----------
    adjacency : np.ndarray
        An adjacency matrix of shape (NxN)
    """
    if not isinstance(adjacency,np.ndarray):
        try: adjacency = np.array(adjacency)
        except Exception as e:
            return e
    degree = np.zeros_like(adjacency)
    np.fill_diagonal(degree, np.sum(adjacency,axis=0))
    return degree

def laplacian(adjacency):
    """
    Computes the Laplacian from a given adjacency matrix 

    Parameters
    ---------- 
    adjacency : np.ndarray
        The adjacency matrix as an array.
    """
    if not isinstance(adjacency,np.ndarray):
        try: adjacency = np.array(adjacency)
        except Exception as e:
            return e
    return degree(adjacency) - adjacency 
