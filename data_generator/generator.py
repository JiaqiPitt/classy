import numpy as np
import anndata as ad

from tools.pdf import linear_pdf_generator,gaussian_pdf_generator 

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
        theta = np.random.uniform(0, 2*np.pi, size=n_samples)  # angular axis
        position  = np.vstack((r, theta)).T 

        if label_bound == None:
            label_bound = (r.max() + r.min())/2

        elif label_bound <= r.min() or label_bound >= r.max():
            raise ValueError('Boundary for points segregation must be within r range.')

        # Assign labels based on radius r
        labels = np.where(r <= 1, 'in', 'out')

        adata = ad.AnnData(position)
        adata.var_names = ['r', 'theta']
        adata.obs_names = [f"Point_{i:d}" for i in range(adata.n_obs)]
        adata.obs['u'] = u  #uniform distributed random numbers
        adata.obs['r'] = r  # radial axis 
        adata.obs['theta'] = theta # angular axis
        adata.obs['Labels'] = labels
        adata.uns['pdf'] =  pdf_dict
        adata.uns['n_samples'] = n_samples
        adata.uns['label_bound'] = label_bound

        return adata
    
    
    