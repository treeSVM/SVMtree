import numpy as np
from Tree import BinaryTree, printTree
from itertools import combinations
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from operator import itemgetter
import pickle


class SVMTree:
    def __init__(self):
        self.x = None
        self.y = None
        self.originalY = None
        self.tree = None
        self.prediction = None

    def fit(self, x, y):
        self.x = x
        self.y = self.__convertYtoLetters(y)
        self.originalY = y
        self.tree = self.__fit()
        return self.tree

    def predict(self, x):
        self.prediction = self.__convertPrediction(self.__predict(self.tree, x))
        return self.prediction

    def getPerformance(self, yTest, nombreDS):
        return self.__getPerformance(self.tree, yTest, nombreDS)

    def printTree(self):
        printTree(self.tree)

    def saveModel(self, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def reloadModel(self, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def __fit(self, tree=None):
        if tree is None:
            chain = self.__generateChain(self.y)
            combination = self.__getCombination(chain)
            bestCombination = self.__chooseBestSVM(combination, x=self.x, y=self.y)
            tree = BinaryTree(bestCombination)
            tree.insertLeft(bestCombination[2][0])
            tree.insertRight(bestCombination[2][1])
            self.__fit(tree.getLeftChild())
            self.__fit(tree.getRightChild())
        else:
            if len(tree.getNodeValue()) == 1:
                return
            else:
                newX, newY = self.__reduceXY(tree.getNodeValue())
                combination = self.__getCombination(tree.getNodeValue())
                bestCombination = self.__chooseBestSVM(combination, x=newX, y=newY)
                tree.setNodeValue(bestCombination)
                tree.insertLeft(bestCombination[2][0])
                tree.insertRight(bestCombination[2][1])
                self.__fit(tree.getLeftChild())
                self.__fit(tree.getRightChild())
        return tree

    # Generates a chain to generate combinations from the 'y' training list
    def __generateChain(self, y):
        return list(set(y))

    # Convert original 'y' training list to letters
    def __convertYtoLetters(self, y):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z']
        newY = []
        chain = self.__generateChain(y)
        lettersChain = []
        for l in range(len(chain)):
            lettersChain.insert(l, letters[l])
        for i in range(len(y)):  # Find a better way to do this to improve performance.
            for j in range(len(lettersChain)):
                if (y[i] == chain[j]):
                    newY.insert(i, lettersChain[j])
        return newY

    # Choose the best SVM of all the possible combinations.
    # We pass x y params because we have to use a different x y value
    def __chooseBestSVM(self, combination, x, y):
        finalResult = []
        for i in range(len(combination)):
            r = self.__getBestSVM(combination=combination[i], x=x, y=y)
            finalResult.insert(i, r)

        finalResult.sort(reverse=True, key=itemgetter(0))
        return finalResult[0]

    # Get the best SVM with the corresponding combination.
    # We pass x y params because we have to use a different x y value
    def __getBestSVM(self, combination, x, y):
        y2 = self.__generateNewY(combination, y)
        v = self.__findBestSVM(x, y2)
        v.insert(2, combination)
        return v

    # Find the best set of parameters to obtain the highest SVM score.
    # We pass x y params because we have to use a different x y value
    def __findBestSVM(self, x, y):
        # Dictionary of possible parameters
        parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 120, 130],
                      'gamma': [0.0001, 0.001, 0.01, 0.1],
                      'kernel': ['linear', 'rbf']}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, cv=5, n_jobs=-1)
        # clf = GridSearchCV(svc, parameters, cv=5)
        clf.fit(x, y)
        return [clf.score(x, y), clf.best_params_, clf]

    # Convert the 'y' of letters into the corresponding tag of a combination.
    def __generateNewY(self, tag, y):
        newY = []
        for i in range(len(y)):
            if y[i] in tag[0]:
                newY.append(tag[0])
                # newY[i] = tag[0]
            else:
                newY.append(tag[1])
        return newY

    # Delete all the rows whose class is not in the tag of a combination
    def __reduceXY(self, tag):
        newX = []
        newY = []
        for i in range(len(self.y)):
            if self.y[i] in tag:
                newY.append(self.y[i])
                newX.append(self.x[i])
        newX = np.array(newX)
        return newX, newY

    # Get the size of the combination partition.
    def __getAmount(self, length):
        if (length % 2 == 0):
            return int(round(length / 2))
        return int(round(length / 2 + 0.5))

    # Get all of the posible combinations for a list of variables (classes in the dataset).
    def __getCombination(self, listOfVariables):
        firstCombinations = np.array(list(combinations(listOfVariables, self.__getAmount(len(listOfVariables)))))
        finalCombinations = []

        if len(firstCombinations) == 2:
            return [listOfVariables]

        for i in range(len(firstCombinations)):  # Find a better way to do this to improve performance.
            finalCombinations.append([])
            for j in range(2):
                finalCombinations[i].append(None)

        for i in range(len(firstCombinations)):
            finalCombinations[i][0] = ''.join(firstCombinations[i])
            finalCombinations[i][1] = ''.join(set(listOfVariables) - set(firstCombinations[i]))

        return finalCombinations

    # Predict the classes of a dataset.
    def __predict(self, tree, x, index=None):
        if tree is not None:
            resultList = []
            if len(tree.getNodeValue()) != 1:
                if index is None:
                    newX = x
                else:
                    newX = x[index]
                prediction = tree.getNodeValue()[3].predict(newX)
                x1, x2 = self.__splitPrediction(prediction, tree.getNodeValue()[2], index)
                r1 = self.__predict(tree.getLeftChild(), x, x1)
                r2 = self.__predict(tree.getRightChild(), x, x2)
                r = r1 + r2
                if index is None:
                    r.sort(key=itemgetter(0))
                    return np.array(r)
                else:
                    return r
            else:
                for i in range(len(index)):
                    resultList.append([index[i], tree.getNodeValue()])
                return resultList

    # Separate the prediction results by both tags
    # We pass y param because we have to use a different y value
    def __splitPrediction(self, y, tag, index=None):
        x1 = []
        x2 = []
        if index is None:
            for i in range(len(y)):
                if y[i] == tag[0]:
                    x1.append(i)
                else:
                    x2.append(i)
        else:
            for i in range(len(y)):
                if y[i] == tag[0]:
                    x1.append(index[i])
                else:
                    x2.append(index[i])
        return x1, x2

    # Change prediction tag by the original ones.
    def __convertPrediction(self, prediction):
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                   'u', 'v', 'w', 'x', 'y', 'z']
        newY = []
        chain = self.__generateChain(self.originalY)
        for i in range(len(prediction)):
            for j in range(len(chain)):
                if prediction[i][1] == letters[j]:
                    newY.append(chain[j])
        return newY

    def __getSupportVectors(self, tree, sv=None, c=[], hms = []):
        if len(tree.getNodeValue()) != 1:
            supportVectors = tree.getNodeValue()[3].best_estimator_.support_vectors_
            cValue = tree.getNodeValue()[1]['C']
            alphas = np.abs(tree.root[3].best_estimator_._dual_coef_)[0]
            if sv is None:
                sv = supportVectors
                c.insert(0, cValue)
                hms.insert(0, self.__hardMarginPercentage(alphas, cValue))
            else:
                np.append(sv, supportVectors)
                c.insert(len(c), cValue)
            self.__getSupportVectors(tree.getLeftChild(), sv, c, hms)
            self.__getSupportVectors(tree.getRightChild(),sv, c, hms)
            return sv, c, hms
        else:
            return sv, c, hms

    def __hardMarginPercentage(self, alphas, c):
        HMS = 0
        for i in range(len(alphas)):
            if alphas[i] == c:
                HMS += 1
        return (HMS / len(alphas)) * 100

    def __getPerformance(self, tree, yTest, nombreDS):

        p = np.array(self.prediction)
        score = np.sum(yTest == p) / len(p)

        results = []
        results.append("Dataset: {}".format(nombreDS))
        results.append("N: {}".format(len(self.x)))
        results.append("Classes: {}".format(len(set(self.originalY))))
        results.append('Score: {}'.format(str(score * 100)))
        sv, C, hms = self.__getSupportVectors(tree)
        results.append('%SV: {}'.format(str((len(sv) / len(self.x)) * 100)))
        results.append('C: {}'.format(C))
        results.append('%HMS: {}'.format(sum(hms) / len(hms)))

        return results