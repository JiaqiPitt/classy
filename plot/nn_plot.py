import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_nn_polar(adata, train_or_test = 'test'):
    """Plots decision boundaries of model predicting on X in comparison to y.
    Source - https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py (modified to use anndata data frame)
    """
    model = adata.uns['nn_model']

    X_train, X_test, Y_train, Y_test, _ = adata.uns['pp_data'].values()

    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X_train, X_test,Y_train, Y_test= X_train.to("cpu"), X_test.to("cpu"), Y_train.to("cpu"), Y_test.to("cpu")

    if train_or_test == 'test':
        X = X_test
        y = Y_test
    else:
        X = X_train
        y = Y_train

    # Setup prediction boundaries and grid
    r_min, r_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    theta_min, theta_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1


    
    grid_r, grid_theta = np.meshgrid(np.linspace(r_min, r_max, 1000), np.linspace(theta_min, theta_max, 1000))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((grid_r.ravel(), grid_theta.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    
    y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(grid_r.shape).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.contourf(grid_theta, grid_r, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    ax.scatter (X[:, 1], X[:,0], c = y, s = 40, cmap = plt.cm.RdYlBu)
    
def visualize_nn_cartesian(adata, train_or_test = 'test'):

    model = adata.uns['nn_model']

    X_train, X_test, Y_train, Y_test, _ = adata.uns['pp_data'].values()

    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X_train, X_test,Y_train, Y_test= X_train.to("cpu"), X_test.to("cpu"), Y_train.to("cpu"), Y_test.to("cpu")

    if train_or_test == 'test':
        X = X_test
        y = Y_test
    else:
        X = X_train
        y = Y_train

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1


    
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(x_min, y_max, 1000))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((grid_x.ravel(), grid_y.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    
    y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(grid_x.shape).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(grid_x, grid_y, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7, extend='both')
    ax.scatter (X[:, 0], X[:,1], c = y, s = 40, cmap = plt.cm.RdYlBu)


def visualize_nn(adata, train_or_test = 'test'):
    # get coordinate type
    _ ,_, _, _, coord = adata.uns['pp_data'].values()

    if coord == 'polar':
        visualize_nn_polar(adata, train_or_test = train_or_test)
    
    elif coord == 'cartesian':
        visualize_nn_cartesian(adata, train_or_test = train_or_test)
    
    else:
        print('Currently only support polar or cartesian coordiante.')
