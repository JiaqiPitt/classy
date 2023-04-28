import matplotlib.pyplot as plt
import numpy as np

from ..classifier import  (neuron, 
                           nn_evaluation, 
                           gp_classifier,
                           gp_evaluation,
                           svm,
                           svm_evaluation,
                           rand_forest,
                           rand_forest_evaluation,
                           xgboost,
                           xgboost_evaluation,
                           knn,
                           knn_evaluation,
)

def compare_models(adata, 
                   
                   ### common factors:
                   use_noise = False,
                   coordinate = 'polar', 
                   test_size = 0.2, 

                   ### random state
                   random_state = None, 
                   print_evaluation_result = True,

                   ### neuron network parameters:
                   act_type = 'sigmoid',
                   loss_fn = 'BCEWithLogitsLoss', 
                   optimizer = 'SGD', 
                   epochs = 10000, 
                   lr = 0.1, 
                   anual_seed = 42, 
                   check_progress = False,
                   ### gaussian process parameters:
                   kernel_type = 'RBF',
                   ### SVM (no new parameter needed)
                   ### random forest parameters:
                   n_estimators  = 1000,
                   max_depth = 10,
                   ### xgboost parameters:
                   num_rounds = 100,
                   params = {'objective': 'binary:logistic',
                      'eval_metric': 'error',
                      'max_depth': 2,
                      'eta': 0.1,
                      'subsample': 0.5,
                      'colsample_bytree': 1,
                      'seed': 42}
                   ### knn (no new parameter needed)
                   ):
    """
    Compare the accuracies given by different classifiers.
    """

    # neuron network
    neuron(adata, 
           coordinate = coordinate, 
           test_size = test_size, 
           random_state = random_state, 
           use_noise = use_noise,
           act_type = act_type,
           loss_fn = loss_fn, 
           optimizer = optimizer, 
           epochs = epochs, 
           lr = lr, 
           anual_seed = anual_seed, 
           check_progress = check_progress)
    
    nn_evaluation(adata, print_evaluation_result = print_evaluation_result)
    
    # Gaussian process 
    gp_classifier(adata, 
                  coordinate = coordinate, 
                  test_size = test_size, 
                  kernel_type = kernel_type, 
                  random_state = random_state, 
                  use_noise = use_noise)
    
    gp_evaluation(adata, print_evaluation_result = print_evaluation_result)

    # SVM
    svm(adata, 
        coordinate = coordinate,
        test_size = test_size, 
        random_state = random_state, 
        use_noise = use_noise)
    
    svm_evaluation(adata, print_evaluation_result = print_evaluation_result)

    # random forest
    rand_forest(adata, 
                coordinate = coordinate, 
                test_size = test_size, 
                n_estimators = n_estimators, 
                max_depth = max_depth, 
                random_state = random_state, 
                use_noise = use_noise)
    
    rand_forest_evaluation(adata, print_evaluation_result = print_evaluation_result)

    # xgboost
    xgboost(adata, 
            coordinate = coordinate,
            test_size = test_size,  
            max_depth = max_depth, 
            random_state = random_state, 
            num_rounds = num_rounds,
            params = params,
            use_noise = use_noise)
    
    xgboost_evaluation(adata, print_evaluation_result = print_evaluation_result)

    # knn
    knn(adata, 
        coordinate = coordinate, 
        test_size = test_size, 
        random_state = random_state, 
        use_noise = use_noise)
    
    knn_evaluation(adata, print_evaluation_result = print_evaluation_result)

    return adata
    
def plot_compare_result(adata, factor = 'noise', print_evaluation_result = False):

    labels = ['neuron network', 'gaussian process', 'svm', 'random forest', 'xgboost', 'knn']

    if factor == 'noise':
        adata1 =  compare_models(adata, use_noise = False, print_evaluation_result = print_evaluation_result) # no noise
        adata2 = compare_models(adata, use_noise = True, print_evaluation_result = print_evaluation_result)# with noise

        accuracy1 = [adata1.uns['nn_evaluation']['Accuracy'], 
                    adata1.uns['gp_evaluation']['Accuracy'],
                    adata1.uns['svm_evaluation']['Accuracy'],
                    adata1.uns['rand_forest_evaluation']['Accuracy'],
                    adata1.uns['xgboost_evaluation']['Accuracy'],
                    adata1.uns['knn_evaluation']['Accuracy']
        ]

        accuracy2 = [adata2.uns['nn_evaluation']['Accuracy'], 
                    adata2.uns['gp_evaluation']['Accuracy'],
                    adata2.uns['svm_evaluation']['Accuracy'],
                    adata2.uns['rand_forest_evaluation']['Accuracy'],
                    adata2.uns['xgboost_evaluation']['Accuracy'],
                    adata2.uns['knn_evaluation']['Accuracy']
        ]
        
        # set bar position
        bar_width = 0.1
        x_pos_1 = np.arange(len(accuracy1))
        x_pos_2 = [x + bar_width for x in x_pos_1]

        # plot bars
        plt.bar(x_pos_1, accuracy1, width=bar_width, label='No noise')
        plt.bar(x_pos_2, accuracy2, width=bar_width, label='With noise')
        plt.xticks([i + bar_width/2 for i in range(len(accuracy1))], labels, rotation=45, ha='right')
    
    elif factor == 'coordinate':
        adata1 = compare_models(adata, coordinate = 'polar', print_evaluation_result = print_evaluation_result)
        adata2 = compare_models(adata, coordinate = 'cartesian', print_evaluation_result = print_evaluation_result)

        accuracy1 = [adata1.uns['nn_evaluation']['Accuracy'], 
                    adata1.uns['gp_evaluation']['Accuracy'],
                    adata1.uns['svm_evaluation']['Accuracy'],
                    adata1.uns['rand_forest_evaluation']['Accuracy'],
                    adata1.uns['xgboost_evaluation']['Accuracy'],
                    adata1.uns['knn_evaluation']['Accuracy']
        ]

        accuracy2 = [adata2.uns['nn_evaluation']['Accuracy'], 
                    adata2.uns['gp_evaluation']['Accuracy'],
                    adata2.uns['svm_evaluation']['Accuracy'],
                    adata2.uns['rand_forest_evaluation']['Accuracy'],
                    adata2.uns['xgboost_evaluation']['Accuracy'],
                    adata2.uns['knn_evaluation']['Accuracy']
        ]

        # set bar position
        bar_width = 0.1
        x_pos_1 = np.arange(len(accuracy1))
        x_pos_2 = [x + bar_width for x in x_pos_1]

        # plot bars
        plt.bar(x_pos_1, accuracy1, width=bar_width, label='Polar coordinate')
        plt.bar(x_pos_2, accuracy2, width=bar_width, label='Cartesian coordinate')
        plt.xticks([i + bar_width/2 for i in range(len(accuracy1))], labels, rotation=45, ha='right')

    
    elif factor == 'test_size':
        adata1 = compare_models(adata, test_size = 0.1, print_evaluation_result = print_evaluation_result)
        adata2 = compare_models(adata, test_size = 0.3, print_evaluation_result = print_evaluation_result)
        adata3 = compare_models(adata, test_size = 0.5, print_evaluation_result = print_evaluation_result)
        
        accuracy1 = [adata1.uns['nn_evaluation']['Accuracy'], 
                    adata1.uns['gp_evaluation']['Accuracy'],
                    adata1.uns['svm_evaluation']['Accuracy'],
                    adata1.uns['rand_forest_evaluation']['Accuracy'],
                    adata1.uns['xgboost_evaluation']['Accuracy'],
                    adata1.uns['knn_evaluation']['Accuracy']
        ]

        accuracy2 = [adata2.uns['nn_evaluation']['Accuracy'], 
                    adata2.uns['gp_evaluation']['Accuracy'],
                    adata2.uns['svm_evaluation']['Accuracy'],
                    adata2.uns['rand_forest_evaluation']['Accuracy'],
                    adata2.uns['xgboost_evaluation']['Accuracy'],
                    adata2.uns['knn_evaluation']['Accuracy']
        ]

        accuracy3 = [adata3.uns['nn_evaluation']['Accuracy'], 
                    adata3.uns['gp_evaluation']['Accuracy'],
                    adata3.uns['svm_evaluation']['Accuracy'],
                    adata3.uns['rand_forest_evaluation']['Accuracy'],
                    adata3.uns['xgboost_evaluation']['Accuracy'],
                    adata3.uns['knn_evaluation']['Accuracy']
        ]

        # set bar position
        bar_width = 0.1
        x_pos_1 = np.arange(len(accuracy1))
        x_pos_2 = [x + bar_width for x in x_pos_1]
        x_pos_3 = [x + bar_width for x in x_pos_2]

        # plot bars
        plt.bar(x_pos_1, accuracy1, width=bar_width, label='10% test data')
        plt.bar(x_pos_2, accuracy2, width=bar_width, label='30% test data')
        plt.bar(x_pos_3, accuracy3, width=bar_width, label='50% test data')
        plt.xticks([i + bar_width for i in range(len(accuracy1))], labels, rotation=45, ha='right')

    else:
        ValueError('currently only support noise, coordinate and test size comparison.')


    plt.ylim(bottom=0.7)
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.title(r'Accuracy with respect to %s'%factor)
    plt.legend()

    # 显示图表
    plt.show()

def compare_nn_act_func(adata,  coordinate = 'cartesian'):

    labels = ['ReLU', 'Sigmoid','Linear']

    adata1 = neuron(adata, coordinate = coordinate, act_type = 'relu').copy()
    adata2 = neuron(adata, coordinate = coordinate, act_type = 'sigmoid').copy()
    adata3 = neuron(adata, coordinate = coordinate, act_type = 'linear').copy()
    nn_evaluation(adata1)
    nn_evaluation(adata2)
    nn_evaluation(adata3)
    accuracy1 = adata1.uns['nn_evaluation']['Accuracy']
    accuracy2 = adata2.uns['nn_evaluation']['Accuracy']
    accuracy3 = adata3.uns['nn_evaluation']['Accuracy']
    
    accuracy = [accuracy1, accuracy2, accuracy3]
    plt.bar(labels, accuracy)
    plt.ylim(bottom=0.7)
    plt.xlabel("Activation function type")
    plt.ylabel("Accuracy")
    plt.title("Neuron network accuracy with repect to activation functoin type")
    plt.show()
