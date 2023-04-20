import numpy as np
import matplotlib.pyplot as plt

from ..tools.utils import get_mapper


def scatter_polar(adata, noise = False):

    if noise:

        theta = adata.obs['theta_noisy']
        r = adata.obs['r_noisy']

    else:
    
        theta = adata.obs['theta']
        r = adata.obs['r']

    # Assign colors based on radius r
    classes = adata.obs['Classes'].tolist()
    mapper = get_mapper()

    colors = []
    for in_out in classes:
        colors.append(mapper[in_out])

    rticks = None if max(r) <= 1 else [i for i in range(1, int(r.max()))]

    # Plot the resulting distribution of (r, theta) as a scatter plot with noise
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(theta, r, s=5, c=colors)
    ax.set_rticks(rticks)
    # Remove the angle scales
    ax.set_thetagrids([0,90,180,270])
    ax.set_rlim(0, np.ceil(r.max())) # Set the limits of the radial axis
    # Remove the grid lines
    ax.grid(visible=False)
    plt.show()


def scatter_cartesian(adata, noise = False):

    if noise:
        
        x = adata.obs['x_noisy']
        y = adata.obs['y_noisy']

    else:
    
        x = adata.obs['x_ori']
        y = adata.obs['y_ori']
    
    # Assign colors based on radius r
    classes = adata.obs['Classes'].tolist()
    mapper = get_mapper()

    colors = []
    for in_out in classes:
        colors.append(mapper[in_out])
    
    # Plot the resulting distribution of (x, y) as a scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s = 5, c = colors)
    plt.axis('equal')
    plt.show()


def scatt(adata, coordinate = 'polar', noise = False):
    
    if coordinate == 'polar':
        scatter_polar(adata, noise = noise)
    
    else:
        scatter_cartesian(adata, noise = noise)