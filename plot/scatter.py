import numpy as np
import matplotlib.pyplot as plt

from ..tools.utils import get_mapper

def scatt(adata):
    
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