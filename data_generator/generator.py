import numpy as np
import anndata as ad

from ..tools.pdf import linear_pdf_generator,gaussian_pdf_generator 

class D:
    
    def __init__(self, n_samples):
        self.n_samples = n_samples
    
    def mc(self, pdf_type, params, label_bound = None):
        """
        Monte Carlo generator for given probability density functions.

        Arguments:
        ---------
            n_samples:  'int'
                        Number of Monte Carlo samples.
            pdf_type: 'str'
                    The type of density distribution function for one dimension
            params: 'dict'        
                    Required parameters for different types of pdf generator.
                    Linear: {'k': k, 'b': b, 'x_right_bound': x_right_bound}
                    Gaussian: {'mu': mu, 'sigma': sigma, 'x_right_bound': x_right_bound}
            label_bound: 'float'
                         Defining the boundary for segregating points. If None, mc function will generate one.
        Returns:
        -------
            data:  'numpy.ndarray'
                n_samples of Monte Carlo data points.
        """
        n_samples = self.n_samples

        if pdf_type not in ['Linear', 'Gaussian']:
            raise ValueError('pdf_type is not included in current package')

        if pdf_type == 'Linear':

            k, b, x_right_bound = params.values()
            pdf, pdf_quantile, x_range = linear_pdf_generator(k, b, x_right_bound)
          
        if pdf_type == 'Gaussian':
            mu, sigma, x_right_bound = params.values()
            pdf, pdf_quantile, x_range = gaussian_pdf_generator(mu, sigma, x_right_bound)

        pdf_dict = {'pdf_type': pdf_type, 'pdf': pdf, 'pdf_quantile': pdf_quantile, 'x_range': x_range}
        
        # Use Monte Carlo method to generate data points
        u = np.random.uniform(size = n_samples) #uniform distributed random numbers
        r = pdf_quantile(u) # radial axis 

        if pdf_type == 'Gaussian':
            r = np.abs(r)

        theta = np.random.uniform(0, 2*np.pi, size=n_samples)  # angular axis
        position  = np.vstack((r, theta)).T 

        if label_bound == None:
            label_bound = (r.max() + r.min())/2

        elif label_bound <= r.min() or label_bound >= r.max():
            raise ValueError('Boundary for points segregation must be within r range.')

        # Assign labels based on radius r
        classes = np.where(r <= 1, 'in', 'out')
        labels = np.where(r <= 1, 1, 0)

        adata = ad.AnnData(position)
        adata.var_names = ['r', 'theta']
        adata.obs_names = [f"Point_{i:d}" for i in range(adata.n_obs)]
        adata.uns['u'] = u  #uniform distributed random numbers
        adata.obs['r'] = r  # radial axis 
        adata.obs['theta'] = theta # angular axis
        adata.obs['Classes'] = classes
        adata.obs['Labels'] = labels
        adata.uns['pdf'] =  pdf_dict
        adata.uns['n_samples'] = n_samples
        adata.uns['label_bound'] = label_bound

        return adata
    
def add_gaussian_noise(adata, loc = 0, scale = 0.2, noise_direction = 'x'):

    r = adata.obs['r']
    theta = adata.obs['theta']

    x_0 = r * np.cos(theta)
    y_0 = r * np.sin(theta)

    noise = np.random.normal(loc=loc, scale = scale, size = adata.uns['n_samples'])

    if noise_direction == 'x':
        x = x_0 + noise
        y = y_0
    
    elif noise_direction == 'y':
        x = x_0
        y = y_0 + noise
    
    else:
        ValueError("noise_direction can either be 'x' or 'y'.")

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    position = position  = np.vstack((r, theta)).T 

    adata.layers['noise_data'] = ad.AnnData(position)
    adata.uns['noise'] = {'loc': loc, 'scale': scale, 'noise_direction': noise_direction}

    return adata
    
 
    