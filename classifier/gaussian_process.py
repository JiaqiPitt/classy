from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn import metrics

import numpy as np

def gp_classifier(adata, coordinate = 'polar', test_size = 0.2, kernel_type = 'RBF', random_state = None, use_noise = False):
    """
    Use Gaussian Process and Laplace Approximation to do binary classification.

    """

    if coordinate == 'polar':
        if use_noise:
            X = adata.layers['noise_data']
        else:
            X = adata.X
    
    elif coordinate == 'cartesian':
        if use_noise:
            X = adata.layers['data_cartesian_noisy']
        else:
            X = adata.layers['data_cartesian']
    
    y = adata.obs['Labels']

     # Segregate the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test, 'Coordinate': coordinate}
    
    gp_classifier = GaussianProcessClassifier()
    
    if kernel_type == 'RBF':
        gp_classifier = GaussianProcessClassifier(kernel=1.0 * RBF(1.0))
        gp_classifier.fit(X_train, Y_train)
    else:
        gp_classifier.fit(X_train, Y_train)

    Y_pred = gp_classifier.predict(X_test)

    adata.uns['gp_result'] = {'data': adata.uns['pp_data'], 'Classifier': gp_classifier, 'Y_prediction': Y_pred}

    
    return adata

def gp_evaluation(adata, print_evaluation_result = True):

    y_test = adata.uns['gp_result']['data']['Y_test']
    y_pred = adata.uns['gp_result']['Y_prediction']

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    if print_evaluation_result:
        print('gaussian process evaluation:')
        print('Confusion matrix:\n', confusion_matrix)
        print('Accuracy:', accuracy_score)
        print('Precision:', precision_score)
        print('Recall:', recall_score)

    adata.uns['gp_evaluation'] = {'Confusion matrix': confusion_matrix, 
                                'Accuracy': accuracy_score, 
                                'Precision': precision_score, 
                                'Recall': recall_score}
    
    return adata
