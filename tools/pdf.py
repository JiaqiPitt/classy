import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erf, erfinv

def linear_pdf_generator(k, b, x_right_bound):
    """
    Generate one dimensional linear probability density function p(x) = nor_factor * (k * x + b), 
    along with pdf_quantile(u) function for the inverse transform method to generate p(x) distribution.
    nor_factor is a normalization factor.
    
    Arguments:
    ---------
        k:  'float'
        b:  'float'
        x_right_bound: 'float'
                       The right boundary of the x-axis.
    Returns:
    -------
        pdf:  'function'
             Linear probability density function p(x). 
        pdf_quantile: 'func'
                    pdf_quantile(u) function for the inverse transform method to generate p(x) distribution.
                    u is uniform distributed random numbers.
        x_range: 'list'
                  The range of x after normalization.
    
    """
    if b < 0:
      raise ValueError("b must be non-negative.")

    if np.logical_and(k < 0, - b/k < x_right_bound):
       
       x_right_bound = - b/k
       nor_factor  = 1/ ( 0.5 * x_right_bound * b)

    else:

      nor_factor = 1/ ( 0.5 * x_right_bound * (k * x_right_bound + 2 * b) )

    pdf = lambda x: nor_factor * (k * x + b)
    pdf_quantile = lambda u: (- b + np.sqrt(b ** 2 + 2 * nor_factor * k * u) )/ (nor_factor * k)
    x_range = [0, x_right_bound]

    return pdf, pdf_quantile, x_range

def gaussian_pdf_generator(mu, sigma, x_right_bound):
    """
    Generate one dimensional Gaussian probability density function p(x) = \frac{1}{sigma \sqrt{2 \pi}} * exp(- 0.5 * \frac{(x-/mu)^2}{\sigma ^ 2}) ,
    along with pdf_quantile(u) function for the inverse transform method to generate p(x) distribution. 
    
    Arguments:
    ---------
        mu:  'float'
             mean of Gaussian distribution.
        sigma:  'float'
             standard deviation of Gaussian distribution.
        x_right_bound: 'float'
                       The right boundary of the x-axis.

    Returns:
    -------
        pdf:  'function'
             Linear probability density function p(x). 
        pdf_quantile: 'func'
                    pdf_quantile(u) function for the inverse transform method to generate p(x) distribution.
                    u is uniform distributed random numbers.
        x_range: 'list'
                  The range of x after normalization.
    
    """
    if x_right_bound <= 0:
        raise ValueError("x right boundary must be positive.")
    
    # Define Gaussian pdf
    pdf = lambda x: (1 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mu)**2 / (2*sigma**2))

    # Define cumulative distribution function for normalized pdf
    pdf_cdf = lambda x: 0.5 * (1 + erf((x-mu) / (sigma*np.sqrt(2))))
    
    # Define the quantile function for the pdf
    pdf_quantile = lambda x: mu + sigma * np.sqrt(2) * erfinv(2*x - 1)

    x_range = [0, x_right_bound]

    return pdf, pdf_quantile, x_range