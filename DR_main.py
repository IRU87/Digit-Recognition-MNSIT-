import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")

from data_loader import *
from simple_ML_functions import *
from tree_models import *
import fully_connected_NN as fcNN
import convolutional_NN as convNN

##train_x, train_y, test_x, test_y = get_MNIST_data()

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################

def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error

#######################################################################
# 3. Support Vector Machine
#######################################################################

def run_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#######################################################################
# 4. Trees & Forest
#######################################################################

def run_tree_on_MNIST():
    '''


    '''
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = make_tree_prediction(train_x, train_y, test_x)
    test_error = compute_test_error_tree(train_x, train_y, test_x, test_y)
    return test_error

def run_random_forest_on_MNIST():
    '''


    '''
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = make_forest_prediction(train_x, train_y, test_x)
    test_error = compute_test_error_forest(train_x, train_y, test_x, test_y)
    return test_error    
#######################################################################
# 5. Neural Networks
#######################################################################

def run_fc_nn():
    np.random.seed(12321)  
    torch.manual_seed(12321)
    fcNN_loss, fcNN_accuracy = fcNN.main()
    return 1-fcNN_accuracy

def run_conv_nn():
    np.random.seed(12321)  
    torch.manual_seed(12321)
    conv_NN_loss, conv_NN_accuracy = convNN.main()
    return 1-conv_NN_accuracy    





#######################################################################
# 6. Overview
#######################################################################

def compare_simple_models():
    print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))
    print('SVM test_error:', run_svm_on_MNIST())
    print('Classifier Tree test_error:', run_tree_on_MNIST())
    print('Random Forest test_error:', run_random_forest_on_MNIST())


    

##if __name__ == '__main__':
##    compare_simple_models()
