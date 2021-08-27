import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from chemsp.utils import gini
import networkx as nx 

from matplotlib.offsetbox import AnchoredText


def signal_plot(coeffs,ylim=(-5,5),linewidth=1,gini=gini,ax=None,sort=False,spines=None,*args, **kwargs):
    """
    Plot a signal plot for the unsorted, signed coefficients of the Fourier expansion. 
    
    Parameters 
    ----------
    coeffs : np.array, (N,)
        The numpy array of expansion coefficients
        
    ylim : tuple, default=(-5,5)
        The y-limits of the plot formatted as (ymin,ymax)
        
    gini : func, default=gini
        A function to compute the gini coefficient of the spectra 
        
    Returns 
    -------
    signal plot, matplotlib.Axes
        The signal plot plotted on axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    ax.set_ylim(*ylim)
    if sort:
        coeffs = np.sort(np.abs(coeffs))
    for x,coeff in enumerate(coeffs):
        ax.plot((x,x),(0,coeff),linewidth=linewidth,c='k',*args, **kwargs)
    if gini is not None:
        ax.annotate(f"Gini Coefficient: {gini(coeffs):.4f}",xy=(0.1,ylim[1]-(ylim[1]/10)))
        
    if spines is not None:
        for spine in spines:
            ax.spines[spine].set_visible(False)
    return ax

def spectrum_plot(coeffs,ylim=(-5,5),gini=gini,ax=None,*args,**kwargs):
    """
    Plots the sorted coefficient spectra on the same plot along with their gini coefficient values
    
    Parameters
    ----------
    coeffs : {representation : np.array(N,)}
        A dictionary with names assigned to each coefficient array of N coefficients. 
        
    ylim : tuple, default=(-5,5)
        The y-limits of the plot formatted as (ymin,ymax)
        
    gini : func, default=gini
        A function to compute the gini coefficient of the spectra 

    Returns 
    -------
    ax : A matplotlib axis figure with all plots
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16,6))
    for name,coeff in coeffs.items():
        coeff = np.sort(np.abs(np.array(coeff)))
        ax.plot(np.arange(len(coeff)), coeff,label=f"{name} ({gini(coeff):.4f})")
    ax.set_xlim(0,len(coeff))
    return ax

def plot_adj(adj,
             node_cmap=matplotlib.cm.get_cmap('Spectral'),
             edge_cmap=matplotlib.cm.get_cmap("Greys"),
             pos=nx.spring_layout,
             title=None,
             ax=None,
             signal=None,
             border_edge_cmap=None,
             return_pos=False,
             *args,
             **kwargs):
    """
    Produce a visualisation of a matrix using networkx.draw. 
    
    Parameters
    ----------
    adj : np.array(N,N)
        The adjacency matrix to be visualised.
        
    node_cmap : matplotlib.cm, default=matplotlib.cm.get_cmap("Spectral")
        The node colourmap used to colour the nodes.
    
    edge_cmap : matplotlib.cm, default=matplotlib.cm.get_cmap("Greys")
        The colormap used to colour edge weights.
    
    pos : func or dict, default=nx.spring_layout
        The network layout function used to compute the positions of the nodes. Alternatively, can explicitly pass dictionary of numpy coordinates to embed
        the vertices.
        
    title : str, default=None
        The string used to title the ax object.
            
    ax : matplotlib.Axes, default=None
        A matplob axis to perform the plot on.
        
    signal : np.array(N,), default=None
        The signal to be projected onto the nodes through face colouring.
        
    border_edge_cmap : dict, default=None
        A colourmap dictionary used to provide border colours to the nodes.
        
    edge_signal
        
    return_pos : bool, default=False
        Whether or not to return the positions dictionary.
        
    Returns 
    -------
    ax : matplotlib.Axes 
        Returns an axis object with the graph plot.
    """
    G = nx.from_numpy_array(adj)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    if callable(pos):
        pos = pos(G)
        
    edge_colors = edge_cmap(weights)
    edge_colors[:,3] = 1-edge_colors[:,0]
    
    if node_cmap is None:
        colors = [(0.5,0.5,0.5,1) for x in range(len(G))]
    else:
        if signal is None:
            colors = node_cmap(np.linspace(0,1,len(G.nodes)))
        else:
            colors = node_cmap(signal)
        
    if border_edge_cmap is not None:
        edgecolors = border_edge_cmap
    else:
        edgecolors = "k"
    
    nx.draw(G,node_color=colors,pos=pos,edgecolors=edgecolors,edge_color=edge_colors,ax=ax, *args, **kwargs)
    
    if title is not None:
        ax.set_title(title,fontsize=24)
    
    if return_pos:
        return ax, pos 
    else:
        return ax


def save(name,directory='/'.join(os.getcwd().split('/')[:-1]) + "/Images",sub_dir=None,*args, **kwargs):
    """
    Custom save function to save images just how I like them.
    
    Parameters
    ----------
    name : str 
        The filename used to save the file
        
    directory : str, default='/'.join(os.getcwd().split('/')[:-1]) + "/Images"
        The directory to save the files into. Defaults to the Images subdirectory in the 
        directory above (assumes this is executed in a code subdirectory.)
        
    sub_dir : str, default=None
        Places the images into a sub directory within images.
    """
    assert name.count('.') <= 1, "Names with multiple periods are poor practice and not supported."
    
    if sub_dir is not None:
        if not os.path.exists('/'.join(os.getcwd().split('/')[:-1])+"/Images/" + f"{sub_dir}"):
            os.mkdir('/'.join(os.getcwd().split('/')[:-1])+"/Images/" + f"{sub_dir}")
    
        fname = directory + "/" + f"{sub_dir}/" + name
    else:
        fname = directory + "/" + name 
        
    if name.split('.')[-1] == ".png" or name.count('.') == 0:
        plt.savefig(fname,dpi=300,bbox_inches="tight",transparent=True, *args, **kwargs)
    elif name.split('.')[-1] == ".pdf":
        plt.savefig(fname,bbox_inches="tight",*args, **kwargs)
    elif name.split('.')[-1] == '.svg':
        plt.savefig(fname, bbox_inches='tight',transparent=True, *args, **kwargs)
    else:
        plt.savefig(fname,bbox_inches="tight",*args,**kwargs)
