import torch
from torch import nn
from sklearn.model_selection import train_test_split
from ..tools.nn_test import accuracy_fn
from sklearn import metrics

def data_split(adata, coordinate = 'polar', test_size = 0.2, random_state = None, use_noise = False):

    if coordinate == 'polar':
        if use_noise:
            X = torch.from_numpy(adata.layers['noise_data']).type(torch.float)
        else:
            X = torch.from_numpy(adata.X).type(torch.float)

    elif coordinate == 'cartesian':
        if use_noise:
            X = torch.from_numpy(adata.layers['data_cartesian_noisy']).type(torch.float)
        else:
            X = torch.from_numpy((adata.layers['data_cartesian'])).type(torch.float)

    Y = torch.tensor(adata.obs['Labels'].values)

    if random_state == None:
        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size = test_size) # 20% test, 80% train
        
    else:                                        
        X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                            Y, 
                                                            test_size = test_size, # 20% test, 80% train
                                                            random_state = random_state) # make the random split reproducible
    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test, 'Coordinate': coordinate}

    return adata


def build_model(adata, act_type = 'sigmoid'):

    class CircleBoundary_ReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=100)
            self.layer_2 = nn.Linear(in_features=100, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
            self.relu = nn.ReLU() # <- add in ReLU activation function
            

        def forward(self, x):
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

    class CircleBoundary_Sigmoid(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=100)
            self.layer_2 = nn.Linear(in_features=100, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
            self.sigmoid = nn.Sigmoid() # <- add in sigmoidal activation function
            

        def forward(self, x):
            return self.layer_3(self.sigmoid(self.layer_2(self.sigmoid(self.layer_1(x)))))
        
    class CircleModel_Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=100)
            self.layer_2 = nn.Linear(in_features=100, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
        
        def forward(self, x):
        
            return self.layer_3(self.layer_2(self.layer_1(x))) 

    
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if act_type == 'relu':
        model = CircleBoundary_ReLU().to(device)
    
    if act_type == 'sigmoid':
        model = CircleBoundary_Sigmoid().to(device)

    if act_type == 'linear':
        model = CircleModel_Linear().to(device)

    adata.uns['nn_model'] = model

    return adata


def train(adata, loss_fn = 'BCEWithLogitsLoss', optimizer = 'SGD', epochs = 10000, lr = 0.1, anual_seed = 42, check_progress = False):

    model = adata.uns['nn_model']
    X_train, X_test, Y_train, Y_test, _ = adata.uns['pp_data'].values()
    # For meet loss function input requirement, change Y's data type to float
    Y_train = Y_train.float()
    Y_test = Y_test.float()

    if loss_fn == 'BCEWithLogitsLoss':
        loss_fn = nn.BCEWithLogitsLoss() # Does not require sigmoid on input
    else: 
        print('Currently only support BCEWithlogitsLoss loss function.')
    
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        print('Currently only support SGD optimizer.')
    
    if anual_seed != None:
        torch.manual_seed(anual_seed)

    for epoch in range(epochs):
        # 1. Forward pass
        Y_logits = model(X_train).squeeze()
        Y_pred = torch.round(torch.sigmoid(Y_logits)) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(Y_logits, Y_train) # BCEWithLogitsLoss calculates loss using logits
        acc = accuracy_fn(y_true = Y_train, 
                        y_pred = Y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits)) # logits -> prediction probabilities -> prediction labels
            # 2. Calcuate loss and accuracy
            test_loss = loss_fn(test_logits, Y_test)
            test_acc = accuracy_fn(y_true = Y_test,
                                y_pred = test_pred)

        # Print out what's happening
        if epoch % 100 == 0:
            if check_progress:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
    
    adata.uns['nn_result'] = {'nn_model': model, 'data': adata.uns['pp_data'], 'Y_prediction': test_pred}
    
    return adata

def nn_evaluation(adata, print_evaluation_result = True):

    y_test = adata.uns['nn_result']['data']['Y_test']
    y_pred = adata.uns['nn_result']['Y_prediction']

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    precision_score = metrics.precision_score(y_test, y_pred)
    recall_score = metrics.recall_score(y_test, y_pred)

    if print_evaluation_result:
        print('neuron network evaluation:')
        print('Confusion matrix:\n', confusion_matrix)
        print('Accuracy:', accuracy_score)
        print('Precision:', precision_score)
        print('Recall:', recall_score)

    adata.uns['nn_evaluation'] = {'Confusion matrix': confusion_matrix, 
                                           'Accuracy': accuracy_score, 
                                           'Precision': precision_score, 
                                           'Recall': recall_score}
    
    return adata


def neuron(adata, 
       coordinate = 'polar', 
       test_size = 0.2, 
       random_state = None, 
       use_noise = False,
       act_type = 'sigmoid',
       loss_fn = 'BCEWithLogitsLoss', 
       optimizer = 'SGD', 
       epochs = 10000, 
       lr = 0.1, 
       anual_seed = 42, 
       check_progress = False):
    
    data_split(adata, coordinate = coordinate, test_size = test_size, random_state = random_state, use_noise = use_noise)
    build_model(adata, act_type = act_type)
    train(adata, loss_fn = loss_fn, optimizer = optimizer, epochs = epochs, lr = lr, anual_seed = anual_seed, check_progress = check_progress)

    return adata





        
    
