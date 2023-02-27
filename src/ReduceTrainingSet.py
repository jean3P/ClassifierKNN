import random

import numpy as np

from KnnClassifier import KnnClassifier


class ReduceTrainingSet:

    @staticmethod
    def condensing(training_set, classifier):

        print("===CONDENSING===")
        # A random element is chosen from the training set.
        randomPos = random.randint(0, len(training_set) - 1)
        randomR = np.reshape(training_set[randomPos], (1, (training_set.shape[1])))
        R = np.array(randomR)
        # The random value is removed from the training set.
        training_set = np.delete(training_set, randomPos, 0)
        changes = True
        while changes:
            changes = False
            sub = 0
            # For each element of the training set
            while sub < (training_set.size / len(training_set[0])):
                subXi = np.reshape(training_set[sub], (1, (training_set.shape[1])))
                # Xi element is classified base on R, with 1NN
                predict = classifier.predict(R, subXi, 1, reduceTrainingSet=True)
                # In case the prediction is not correct
                if predict != subXi[0][training_set.shape[1] - 1]:
                    # That element Xi is added to the list R. The list R will grow.
                    R = np.vstack((R, training_set[sub]))
                    training_set = np.delete(training_set, sub, 0)
                    changes = True
                sub += 1
        return R

    @staticmethod
    def editing(training_set, classifier):
        print("===EDITING===")
        sub = 0
        # List that will store the positions of the items to be deleted
        markForDeletion = []
        while sub < (training_set.size / len(training_set[0])):
            # The Xi to be classified is selected
            subXi = np.reshape(training_set[sub], (1, (training_set.shape[1])))
            # The training subset is prepared (contains all elements except Xi to be classified).
            trainingSubSet = np.delete(training_set, sub, 0)
            # Xi element is classified base on training subset, with 3NN
            predict = classifier.predict(trainingSubSet, subXi, 3, reduceTrainingSet=True)
            # In case the prediction is not correct
            if predict != subXi[0][len(training_set[0]) - 1]:
                # Position Xi to be deleted is marked
                markForDeletion.append(sub)
            sub += 1
        # All positions that were marked are deleted
        R = np.delete(training_set, markForDeletion, 0)
        return R
