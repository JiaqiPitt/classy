import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

def visualize_xgboost_polar(adata, train_or_test = 'test'):

    X_train, X_test, Y_train, Y_test, _ = adata.uns['pp_data'].values()

    if train_or_test == 'train':
        X_set, y_set =  X_train, Y_train
    else:
        X_set, y_set =  X_test, Y_test

    classifier = adata.uns['xgboost_result']['Classifier']


    grid_r, grid_theta = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

    X = np.column_stack((grid_r.ravel(), grid_theta.ravel()))
    dtest = xgb.DMatrix(X)
    y = np.where(classifier.predict(dtest) > 0.5, 1, 0)

    # Polar plot

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)


    ax.contourf(grid_theta, grid_r, y.reshape(grid_r.shape),
                cmap=plt.cm.RdYlBu, alpha=0.7)
    
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 1], X_set[y_set == j, 0],
                    s = 40, cmap = plt.cm.RdYlBu, label = j)
        
def visualize_xgboost_cartesian(adata, train_or_test = 'test'):

    X_train, X_test, Y_train, Y_test, _ = adata.uns['pp_data'].values()

    if train_or_test == 'train':
        X_set, y_set =  X_train, Y_train
    else:
        X_set, y_set =  X_test, Y_test

    classifier = adata.uns['xgboost_result']['Classifier']


    grid_x, grid_y = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    
    X = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dtest = xgb.DMatrix(X)
    y = np.where(classifier.predict(dtest) > 0.5, 1, 0)

    # Polar plot

    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax.contourf(grid_x, grid_y, y.reshape(grid_x.shape),
                cmap=plt.cm.RdYlBu, alpha=0.7)
    
    for i, j in enumerate(np.unique(y_set)):
        ax.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    s = 40, cmap = plt.cm.RdYlBu, label = j)

def visualize_xgboost(adata, train_or_test = 'test'):

    # get coordinate type
    _ ,_, _, _, coord = adata.uns['pp_data'].values()

    if coord == 'polar':
        visualize_xgboost_polar(adata, train_or_test = train_or_test)
    
    elif coord == 'cartesian':
        visualize_xgboost_cartesian(adata, train_or_test = train_or_test)
    
    else:
        print('Currently only support polar or cartesian coordiante.')