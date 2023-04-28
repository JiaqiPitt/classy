import numpy as np
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
        adata.obs_names = [f"Point_{i:d}" for i in range(adata.n_obs)]
        adata.uns['u'] = u  #uniform distributed random numbers
        adata.obs['r'] = r  # radial axis 
        adata.obs['theta'] = theta # angular axis
        adata.obs['Classes'] = classes
        adata.obs['Labels'] = labels
        adata.uns['pdf'] =  pdf_dict
        adata.uns['n_samples'] = n_samples
        adata.uns['label_bound'] = label_bound
        adata.uns['generate_method'] = 'mc'

        return adata

    def generate_polynomial_boundary_example(self, random_seed = 0):

        """
        Generate two clusters of data points whose boudary is a polynomial function.
        """

        np.random.seed(random_seed)

        n_samples = self.n_samples
        x1 = np.random.uniform(low=-10, high=10, size=n_samples)
        x2 = np.random.uniform(low=-10, high=10, size=n_samples)
        X = np.vstack([x1, x2]).T

        # add label based on polynomial functions
        labels = np.zeros(n_samples)
        for i in range(n_samples):
            if x2[i] > x1[i]**3 - 6*x1[i]**2 + 4*x1[i] + 5:
                labels[i] = 1

        classes = np.where(labels == 1, 'Cluster 1', 'Cluster 2')

        adata = ad.AnnData(X)
        adata.obs_names = [f"Point_{i:d}" for i in range(adata.n_obs)]
        adata.obs['x_ori'] = x1
        adata.obs['y_ori'] = x2
        adata.obs['Classes'] = classes
        adata.obs['Labels'] = labels
        adata.uns['n_samples'] = n_samples
        adata.uns['generate_method'] = 'polynomial_example'

        adata_new = transfer_cartesian_to_polar(adata)

        return adata_new
    
    def generate_gussian_clusters_example(self):

        """
        Generate two clusters of gaussian distributed data points.
        """
        n_samples_tot = self.n_samples
        n_samples = int(self.n_samples / 2)

        # define the mean value for each cluster also the covariant matrices
        mean1 = [0, 0]
        cov1 = [[1, 0.5], [0.5, 1.5]]

        mean2 = [2, 2]
        cov2 = [[1.5, 0], [0, 0.5]]

        # generate data points
        cluster1 = np.random.multivariate_normal(mean1, cov1, n_samples)
        cluster2 = np.random.multivariate_normal(mean2, cov2, n_samples)

        # assign labels
        labels1 = np.zeros(n_samples)
        labels2 = np.ones(n_samples)

        # combine two clusters
        data = np.vstack((cluster1, cluster2))
        labels = np.concatenate((labels1, labels2))
        classes = np.where(labels == 1, 'Cluster 1', 'Cluster 2')

        adata = ad.AnnData(data)
        adata.obs_names = [f"Point_{i:d}" for i in range(adata.n_obs)]
        adata.obs['x_ori'] = data[:, 0]
        adata.obs['y_ori'] = data[:, 1]
        adata.obs['Classes'] = classes
        adata.obs['Labels'] = labels
        adata.uns['n_samples'] = n_samples_tot
        adata.uns['generate_method'] = 'gaussian_example'

        adata_new = transfer_cartesian_to_polar(adata)

        return adata_new


    
def add_gaussian_noise(adata, loc = 0, scale = 3, noise_direction = 'polar'):

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
    elif noise_direction == 'polar':
        r = r + noise
    
    else:
        ValueError("noise_direction can either be 'x', 'y'or 'polar'. ")

    if noise_direction != 'polar':
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
    
    position = position  = np.vstack((r, theta)).T 

    adata.layers['noise_data'] = position
    adata.obs['r_noisy'] = r
    adata.obs['theta_noisy'] = theta
    adata.uns['noise'] = {'loc': loc, 'scale': scale, 'noise_direction': noise_direction}

    return adata

def transfer_polar_to_cartesian(adata):
    
    original_data = adata.X
    x_ori = original_data[:,0] * np.cos(original_data[:,1])
    y_ori = original_data[:,0] * np.sin(original_data[:,1])
    data_cartesian = np.stack((x_ori, y_ori), axis = 1)

    # x_ori = original_data[]
    if 'noise_data' in adata.layers.keys():
        noise_data = adata.layers['noise_data']
        x_noisy = noise_data[:,0] * np.cos(noise_data[:,1])
        y_noisy = noise_data[:,0] * np.sin(noise_data[:,1])
        data_cartesian_noisy = np.stack((x_noisy, y_noisy), axis = 1)

    adata.layers['data_cartesian'] = data_cartesian
    adata.layers['data_cartesian_noisy'] = data_cartesian_noisy
    adata.obs['x_ori'] = x_ori
    adata.obs['y_ori'] = y_ori
    adata.obs['x_noisy'] = x_noisy
    adata.obs['y_noisy'] = y_noisy

    return adata

def transfer_cartesian_to_polar(adata):

    original_data = adata.X

    r = np.sqrt(original_data[:, 0] ** 2 + original_data[:, 1] ** 2)
    theta = np.arctan2(original_data[:, 1], original_data[:, 0])

    position  = np.vstack((r, theta)).T 

    adata_transfered = ad.AnnData(position)
    adata_transfered.layers['data_cartesian'] = adata.X
    adata_transfered.obs_names = adata.obs_names
    adata_transfered.obs['Classes'] = adata.obs['Classes']
    adata_transfered.obs['Labels'] = adata.obs['Labels']
    adata_transfered.uns['n_samples'] = adata.uns['n_samples']
    adata_transfered.uns['generate_method'] = adata.uns['generate_method']

    adata_transfered.obs['r'] = r
    adata_transfered.obs['theta'] = theta

    return adata_transfered
    