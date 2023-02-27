import statistics
import numpy as np


def get_manhattan_distance(training_data_point, test_data_point, reduceTrainingSet):
    dist = 0
    if reduceTrainingSet:
        for i in range(len(training_data_point) - 1):
            dist = dist + abs(training_data_point[i] - test_data_point[0][i])

    else:
        for i in range(len(training_data_point) - 1):
            dist = dist + abs(training_data_point[i] - test_data_point[i])
    manhattan_dist = dist
    return manhattan_dist


def get_euclidean_distance(training_data_point, test_data_point, reduceTrainingSet):
    dist = 0
    if reduceTrainingSet:
        for i in range(len(training_data_point) - 1):
            dist = dist + (training_data_point[i] - test_data_point[0][i]) ** 2

    else:
        for i in range(len(training_data_point) - 1):
            dist = dist + (training_data_point[i] - test_data_point[i]) ** 2
    euclidean_dist = np.sqrt(dist)
    return euclidean_dist


class KnnClassifier:

    # initiating the parameters
    def __init__(self, distance_metric):

        self.distance_metric = distance_metric

    # getting the distance metric
    def get_distance_metric(self, training_data_point, test_data_point, reduceTrainingSet):

        if self.distance_metric == 'euclidean':
            return get_euclidean_distance(training_data_point, test_data_point, reduceTrainingSet)

        elif self.distance_metric == 'manhattan':
            return get_manhattan_distance(training_data_point, test_data_point, reduceTrainingSet)

    # getting the nearest neighbors
    def nearest_neighbors(self, x_train, test_data, k, reduceTrainingSet):

        distance_list = []

        for training_data in x_train:
            # print(len(training_data))
            distance = self.get_distance_metric(training_data, test_data, reduceTrainingSet)
            # a set of the training value (a row) with its corresponding distance
            distance_list.append((training_data, distance))

        # sorting list from minimum distance to maximum distance
        # sort using the distance value
        # get a list with an ascending order
        distance_list.sort(key=lambda x: x[1])

        neighbors_list = []

        # try to find the k first values
        for j in range(k):
            # if k is equals to five trends five times in order to find five nearest neighbors
            # get the first value
            # print(k)
            # print(len(distance_list))
            neighbors_list.append(distance_list[j][0])

        return neighbors_list

    # what it`s the class that is the majority?
    # predict the class of the new data point:
    def predict(self, x_train, test_data, k, reduceTrainingSet):
        neighbors = self.nearest_neighbors(x_train, test_data, k, reduceTrainingSet)
        # get the label
        for data in neighbors:
            # [-1] represent the last column value, assuming that the label is in the last column
            label = [data[-1]]

        # return the most common data point (which is repeated the most number of times) from discrete or nominal data
        predicted_class = statistics.mode(label)

        return predicted_class
