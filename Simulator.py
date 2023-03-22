import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

class Simulator:
    
    def __init__(self, n_samples):
        self.n_samples = n_samples
    
    def MC_generator(n_samples, pdf_type, params):
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
        Returns:
        -------
            data:  'numpy.ndarray'
                n_samples of Monte Carlo data points.
        """
        if pdf_type not in ['Linear', 'Gaussian']:
            raise ValueError('pdf_type is not included in current package')
        if pdf_type == 'Linear':

            k, b, x_right_bound = params.values()
            pdf, pdf_quantile, x_range = linear_pdf_generator(k, b, x_right_bound)
          
        if pdf_type == 'Gaussian':
            mu, sigma, x_right_bound = params.values()
            pdf, pdf_quantile, x_range = gaussian_pdf_generator(mu, sigma, x_right_bound)
        
        return pdf, pdf_quantile, x_range