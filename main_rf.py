import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from  sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

"""
    Training and evaluating Random Forest classification metrics using a shuffled dataset in 5 folds of data
"""
N_FOLDS = 5
for i in range(N_FOLDS):

    current_fold = str(i + 1)
    train_fn = '[YOUR TRAIN DATA FILE PATH HERE]'
    validation_fn = '[YOUR VALIDATION DATA FILE PATH HERE]'
    test_fn = '[YOUR TEST DATA FILE PATH HERE]'
    target_train_fn = '[YOUR TRAIN LABELS FILE PATH HERE]'
    target_validation_fn = '[YOUR VALIDATION LABELS FILE PATH HERE]'
    target_test_fn = '[YOUR TEST LABELS FILE PATH HERE]'

    # loading the data already splitted
    x_train = np.load(train_fn)
    x_validation = np.load(validation_fn)
    x_test = np.load(test_fn)
    y_train = np.argmax(np.load(target_train_fn),axis=1)
    y_validation = np.argmax(np.load(target_validation_fn),axis=1)
    y_test = np.argmax(np.load(target_test_fn), axis=1)

    # parameter definition for tuning
    tuned_parameters = {'n_estimators': [200,300,400,500],
    'max_depth': [None, 20,40,60,80,100]}

    # model creation
    clf = RandomForestClassifier()

    # merging train and validation test for the optimization step
    x_train_validation = np.concatenate((x_train,x_validation),axis=0)
    y_train_validation = np.concatenate((y_train, y_validation), axis=0)

    # split_index contains -1 indicating the sample is for training; otherwise 0 for validation
    split_index = [-1 if x in range(len(x_train)) else 0 for x in range(len(x_train_validation))]

    # using the split_index list to define the split criteria for the cross-validation process during the model optimization
    ps = PredefinedSplit(test_fold = split_index)

    # Optimization step. Setting n_jobs = -1 enables parallel execution on all processors
    grid_search = GridSearchCV(estimator = clf, param_grid = tuned_parameters, cv = ps, n_jobs =-1, verbose = 2)
    # training of the random forest classifier
    grid_search.fit(x_train_validation, y_train_validation)
    # test predictions
    y_pred = grid_search.predict(x_test)
    print("Metrics fold ", current_fold)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("F-score: ", metrics.f1_score(y_test, y_pred, average='macro'))
    print("K-score: ", metrics.cohen_kappa_score(y_test, y_pred))