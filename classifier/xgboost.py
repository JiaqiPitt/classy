import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


def xgboost(adata, 
            test_size = 0.2,  
            max_depth = 10, 
            random_state = None, 
            num_rounds = 100,
            params = {'objective': 'binary:logistic',
                        'eval_metric': 'error',
                        'max_depth': 2,
                        'eta': 0.1,
                        'subsample': 0.5,
                        'colsample_bytree': 1,
                        'seed': 42},
            use_noise = False
            ):
    if use_noise:
        X = adata.layers['noise_data']
    else:
        X = adata.X

    
    y = adata.obs['Labels']

    # Segregate the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test, 'params': params}

    # Train the XGBoost classifier
    dtrain = xgb.DMatrix(X_train, label = Y_train)
    classifier = xgb.train(params, dtrain, num_rounds)

    # Make predictions on the test set
    dtest = xgb.DMatrix(X_test)
    Y_pred = np.where(classifier.predict(dtest) > 0.5, 1, 0)

    adata.uns['xgboost_result'] = {'data': adata.uns['pp_data'], 'Classifier': classifier, 'Y_prediction': Y_pred}

    return adata

    
def xgboost_evaluation(adata):

    y_test = adata.uns['xgboost_result']['data']['Y_test']
    y_pred = adata.uns['xgboost_result']['Y_prediction']

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)

    print('Confusion matrix:\n', confusion_matrix)
    print('Accuracy:', accuracy_score)
    print('Precision:', precision_score)
    print('Recall:', recall_score)

    adata.uns['xgboost_evluation'] = {'Confusion matrix': confusion_matrix, 
                                'Acuracy': accuracy_score, 
                                'Precision': precision_score, 
                                'Recall': recall_score}
    
    return adata