from datetime import datetime

import numpy as np
import pandas as pd
import os

from sklearn.metrics import accuracy_score

import Helper
from Helper import reduce_training_set
from KnnClassifier import KnnClassifier
from ReduceTrainingSet import ReduceTrainingSet



dataset_path = '../resources/mnist_small_knn'

reduceDatasetDirectoryCondensing = Helper.reduceDatasetDirectoryCondensing
reduceDatasetDirectoryEditing = Helper.reduceDatasetDirectoryEditing

distancesMetrics = Helper.distancesMetrics
reduceTrainingAlgorithms = Helper.reduceTrainingAlgorithms
namefileReducingDataset = Helper.namefileReducingDataset

if (os.listdir(dataset_path)):
    datasetTrain = pd.read_csv('../resources/mnist_small_knn/train.csv')
    Y_train = datasetTrain.take([0], axis=1)
    datasetTrain.drop(datasetTrain.columns[[0]], axis=1, inplace=True)
    X_train = datasetTrain.to_numpy()
    Y_train = Y_train.to_numpy()

    # Labels are concatenated at the end
    totalTrainingSet = np.append(X_train, Y_train, axis=1)
    print('== Value X_train: ', X_train.size)

    files = Helper.get_path_files()
    datasetTest = pd.read_csv('../resources/mnist_small_knn/test.csv')
    Y_test = datasetTest.take([0], axis=1)
    datasetTest.drop(datasetTest.columns[[0]], axis=1, inplace=True)
    X_test = datasetTest.to_numpy()
    Y_test = Y_test.to_numpy()

    X_train = totalTrainingSet

    accuracy_vals = []
    # start_time = datetime.now()
    KValues = Helper.KValues

for distance in distancesMetrics:
    classifier = KnnClassifier(distance_metric=distance)
    print('==== ' + distance.capitalize() + ' ====')
    for k_n in KValues:
        print(" == For K: ", k_n)
        start_time = datetime.now()
        X_test_size = X_test.shape[0]
        # y_pred = is the value predicted for my model
        y_pred = []
        # try to find the prediction for all test data
        for i in range(X_test_size):
            prediction = classifier.predict(X_train, X_test[i], k=k_n, reduceTrainingSet=False)
            y_pred.append(prediction)
        # y_true = are the true values
        y_true = Y_test
        # the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
        y_true = [float(y_true) for y_true in y_true]
        accuracy = accuracy_score(y_true, y_pred)
        accuracy_vals.append(accuracy * 100)
        end_time = datetime.now()
        print("         Best value (%): ", np.max(accuracy) * 100)
        print('         Duration of Classification: {}'.format((end_time - start_time).total_seconds()))
        print('=======')
