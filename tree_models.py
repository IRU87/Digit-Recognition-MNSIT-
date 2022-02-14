import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


######### Decision Tree

def make_tree_prediction(x_train,y_train, x_test):
    tree_model = DecisionTreeClassifier(random_state=1)
    tree_model.fit(x_train,y_train)
    y_test_predictions = tree_model.predict(x_test)
    return y_test_predictions

def compute_test_error_tree(x_train,y_train, x_test, y_test):
    y_test_pred= make_tree_prediction(x_train,y_train, x_test)
    return 1 - np.mean(y_test_pred == y_test)
    


######### Random Forest

def make_forest_prediction(x_train,y_train, x_test, n_estim = 30):
    tree_model = RandomForestClassifier(n_estim, random_state=1)
    tree_model.fit(x_train,y_train)
    y_test_predictions = tree_model.predict(x_test)
    return y_test_predictions

def compute_test_error_forest(x_train,y_train, x_test, y_test):
    y_test_pred= make_forest_prediction(x_train,y_train, x_test)
    return 1 - np.mean(y_test_pred == y_test)
