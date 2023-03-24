import torch
from torch import nn
from sklearn.model_selection import train_test_split
from ..tools.nn_test import accuracy_fn

def data_split(adata, test_size = 0.2, random_state = None):

    X = torch.from_numpy(adata.X).type(torch.float)
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
    adata.uns['pp_data'] = {'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test}

    return adata


def build_model(adata):

    class CircleBoundary(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=2, out_features=100)
            self.layer_2 = nn.Linear(in_features=100, out_features=100)
            self.layer_3 = nn.Linear(in_features=100, out_features=1)
            self.relu = nn.ReLU() # <- add in ReLU activation function
            

        def forward(self, x):
        # Intersperse the ReLU activation function between layers
            return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    # Make device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CircleBoundary().to(device)

    adata.uns['nn_model'] = model

    return adata


def train(adata, loss_fn = 'BCEWithLogitsLoss', optimizer = 'SGD', epochs = 10000, lr = 0.1, anual_seed = None):

    model = adata.uns['nn_model']
    X_train, X_test, Y_train, Y_test = adata.uns['pp_data'].values()
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
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}%")
        






        
    
