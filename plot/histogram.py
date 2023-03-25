import numpy as np
import matplotlib.pyplot as plt

def histo(adata):

    n_samples = adata.uns['n_samples']
    pdf = adata.uns['pdf']['pdf']
    pdf_type = adata.uns['pdf']['pdf_type']
    r = adata.obs['r']
    

    # Calculate appropriate bin size based on Freedman-Diaconis rule
    iqr = np.percentile(r, 75) - np.percentile(r, 25)
    h = 2 * iqr / (len(r) ** (1/3))
    bins = int((r.max() - r.min()) / h)

    x = np.linspace(r.min(), r.max(), 10000) # For theoretical distribution plot

    if pdf_type == 'Gaussian':
       pdf_line = 2 * pdf(x)

    else:
       pdf_line = pdf(x)

    # Plot the histogram of the generated samples
    plt.hist(r, bins=bins, density=True, alpha=0.5, label='Generated') 
    plt.plot(x, pdf_line, 'r-', label = 'True')
    plt.legend()
    plt.show()