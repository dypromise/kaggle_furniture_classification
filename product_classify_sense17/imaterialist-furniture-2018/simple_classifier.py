import numpy as np
from sklearn.datasets import load_digits
import time
from sklearn import metrics
import pickle as pickle
import pandas as pd


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                               fit_intercept=True, intercept_scaling=1, class_weight=None,
                               random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                               verbose=0, warm_start=False, n_jobs=1)
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                        max_features=None, random_state=None, max_leaf_nodes=None,
                                        min_impurity_decrease=0.0, min_impurity_split=None,
                                        class_weight=None, presort=False)
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10,
                        100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'],
                gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


def read_data(preds_list):
    from sklearn.cross_validation import train_test_split

    for i, pred in enumerate(preds_list):
        arr = np.load(pred)
        if(i == 0):
            idxs = np.array(arr[:, 0], dtype='int32').reshape((-1, 1))
            res = np.array(arr[:, 1:], dtype='float32')
        else:
            res += arr[:, 1:]

    res /= (i+1)
    labels = (np.argmax(res, axis=1)+1).reshape((-1, 1))

    res = np.concatenate((idxs, labels), axis=1)

    data = np.loadtxt(data_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, y_train, X_test, y_test


def main():
    data_file = ''
    thresh = 0.5
    model_save_file = None
    model_save = {}

    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
    classifiers = {
        'LR': logistic_regression_classifier,
        'SVM': svm_classifier,
        'SVMCV': svm_cross_validation,
        'RF': random_forest_classifier,
        'DT': decision_tree_classifier,
        'GBDT': gradient_boosting_classifier
    }

    print('reading training and testing data...')
    train_x, train_y, test_x, test_y = read_data(data_file)

    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        print('precision: %.3f%%, recall: %.3f%%' %
              (100 * precision, 100 * recall))
        accuracy = metrics.accuracy_score(test_y, predict)
        print('accuracy: %.3f%%' % (100 * accuracy))

    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))


if(__name__ == '__main__'):
    main()

avg_ensamble(p_list, test_whole_file,
             os.path.join(data_root, 'ensamble.csv'))
