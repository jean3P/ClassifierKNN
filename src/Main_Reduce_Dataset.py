from datetime import datetime

import numpy as np
import pandas as pd
import os

import Helper
from Helper import reduce_training_set
from KnnClassifier import KnnClassifier
from ReduceTrainingSet import ReduceTrainingSet

datasetTrain = pd.read_csv('../resources/mnist_small_knn/train.csv')

reduceDatasetDirectoryCondensing = Helper.reduceDatasetDirectoryCondensing
reduceDatasetDirectoryEditing = Helper.reduceDatasetDirectoryEditing

distancesMetrics = Helper.distancesMetrics
reduceTrainingAlgorithms = Helper.reduceTrainingAlgorithms
namefileReducingDataset = Helper.namefileReducingDataset

if not (os.listdir(reduceDatasetDirectoryCondensing) and (os.listdir(reduceDatasetDirectoryEditing))):
    Y_train = datasetTrain.take([0], axis=1)
    datasetTrain.drop(datasetTrain.columns[[0]], axis=1, inplace=True)
    X_train = datasetTrain.to_numpy()
    Y_train = Y_train.to_numpy()

    # Labels are concatenated at the end
    totalTrainingSet = np.append(X_train, Y_train, axis=1)
    print('== Value X_train: ', X_train.size)
    for distance in distancesMetrics:
        classifier = KnnClassifier(distance_metric=distance)
        print('==== ' + distance.capitalize() + ' ====')
        for reduceTraining in reduceTrainingAlgorithms:
            start_time = datetime.now()
            reduceTrainingSet = Helper.reduce_training_set(reduceTraining, classifier, totalTrainingSet)
            end_time = datetime.now()
            print('== New Value X_train after using Reduction Algorithm: ', reduceTrainingSet.size)
            print('Duration of Reduction: {}'.format((end_time - start_time).total_seconds()))
            print(" ")
            if reduceTraining == 'condensing':
                np.savetxt((reduceDatasetDirectoryCondensing + '/' + distance + '_' + namefileReducingDataset), reduceTrainingSet,
                           delimiter=',')
            else:
                np.savetxt((reduceDatasetDirectoryEditing + '/' + distance + '_' + namefileReducingDataset), reduceTrainingSet,
                           delimiter=',')
