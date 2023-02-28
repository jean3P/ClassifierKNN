from ReduceTrainingSet import ReduceTrainingSet

reduceDatasetDirectoryCondensing = "../resources/reduceDataset/condensing"
reduceDatasetDirectoryEditing = "../resources/reduceDataset/editing"
distancesMetrics = ['euclidean', 'manhattan']
namefileReducingDataset = 'X_train.csv'
reduceTrainingAlgorithms = ['condensing', 'editing']
KValues = [1, 3, 5, 10, 15]


def get_path_files():
    euclideanCondensingXtrain = reduceDatasetDirectoryCondensing + '/' + distancesMetrics[
        0] + '_' + namefileReducingDataset
    manhattanCondensingXtrain = reduceDatasetDirectoryCondensing + '/' + distancesMetrics[
        1] + '_' + namefileReducingDataset
    euclideanEditingXtrain = reduceDatasetDirectoryEditing + '/' + distancesMetrics[0] + '_' + namefileReducingDataset
    manhattanEditingXtrain = reduceDatasetDirectoryEditing + '/' + distancesMetrics[1] + '_' + namefileReducingDataset
    path_files = [euclideanCondensingXtrain, manhattanCondensingXtrain, euclideanEditingXtrain, manhattanEditingXtrain]
    return path_files


def reduce_training_set(reduce, classifier, X_train):
    if reduce == 'condensing':
        return ReduceTrainingSet.condensing(X_train, classifier)
    else:
        return ReduceTrainingSet.editing(X_train, classifier)
