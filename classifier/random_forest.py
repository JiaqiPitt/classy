from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


def rand_forest(adata, test_size = 0.2, n_estimators = 1000, max_depth = 10, random_state = None):

    X = adata.X
    y = adata.obs['Labels']

    # Segregate the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

    # Train the random forest classifier
    classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth =  max_depth, random_state = random_state)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    adata.uns['rand_forest_result'] = {'data': adata.uns['pp_data'], 'Classifier': classifier, 'Y_prediction': Y_pred}

    return adata

def rand_forest_evaluation(adata):

    y_test = adata.uns['rand_forest_result']['data']['Y_test']
    y_pred =  adata.uns['rand_forest_result']['Y_prediction']

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)

    print('Confusion matrix:\n', confusion_matrix)
    print('Accuracy:', accuracy_score)
    print('Precision:', precision_score)
    print('Recall:', recall_score)

    adata.uns['rand_forest_evaluation'] = {'Confusion matrix': confusion_matrix, 
                                           'Acuracy': accuracy_score, 
                                           'Precision': precision_score, 
                                           'Recall': recall_score}
    
    return adata




