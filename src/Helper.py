from ReduceTrainingSet import ReduceTrainingSet


def reduce_training_set(reduce, classifier, X_train):
    if reduce == 'condensing':
        return ReduceTrainingSet.condensing(X_train, classifier)
    else:
        return ReduceTrainingSet.editing(X_train, classifier)