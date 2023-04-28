from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import metrics

def knn(adata, coordinate = 'polar', test_size = 0.2, random_state = None, use_noise = False):
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
    
    else:
        print('Either polar coordinate or cartesian coordinate is available.')

    y = adata.obs['Labels']

    # Segregate the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test, 'Coordinate': coordinate}
    
    # Create a KNN classifier with k=3
    classifier = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier on the training set
    classifier.fit(X_train, Y_train)

    # Use the classifier to predict the test set
    Y_pred = classifier.predict(X_test)

    adata.uns['knn_result'] = {'data': adata.uns['pp_data'], 'Classifier': classifier, 'Y_prediction': Y_pred}

def knn_evaluation(adata, print_evaluation_result = True):

    y_test = adata.uns['knn_result']['data']['Y_test']
    y_pred =  adata.uns['knn_result']['Y_prediction']

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)
    if print_evaluation_result:

        print('knn evaluation:')
        print('Confusion matrix:\n', confusion_matrix)
        print('Accuracy:', accuracy_score)
        print('Precision:', precision_score)
        print('Recall:', recall_score)

    adata.uns['knn_evaluation'] = {'Confusion matrix': confusion_matrix, 
                                'Accuracy': accuracy_score, 
                                'Precision': precision_score, 
                                'Recall': recall_score}
    
    return adata