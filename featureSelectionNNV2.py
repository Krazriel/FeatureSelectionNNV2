import numpy as np
import collections as co
import copy as cp

#Set up data, features and labels sets
smallDataSet = np.loadtxt('data/CS205_SP_2022_SMALLtestdata__38.txt')
largeDataSet = np.loadtxt('data/CS205_SP_2022_Largetestdata__3.txt')

smallDataLabels = smallDataSet[:, 0]
largeDataLabels = largeDataSet[:, 0]

#Nearest Neighbors Classifier
def nearestNeighbors(target, data, labels):
    nearestDist = float('inf')
    nearestLabel = 0
    for i in range(len(data)):
        if np.all(target == data[i]):
            continue
        if np.linalg.norm(target - data[i]) < nearestDist:
            nearestDist = np.linalg.norm(target - data[i])
            nearestLabel = labels[i]
    return nearestLabel

#Cross Validator
def crossValidation(data, labels):
    pred = []
    for i in range(len(data)):
        test = data[i]
        pred.append(nearestNeighbors(test, data, labels))
    pred = np.array(pred)
    stats = np.sum(pred == labels) / len(labels)
    return stats

#Forward Selection
def forwardSelection(data, labels):
    defaultRate = 100 * co.Counter(labels).get(max(co.Counter(labels))) / len(labels)
    print('Using Features :  D  : accuracy = {:0.1f}%'.format(defaultRate))
    featureList = []

    while(len(featureList) != len(data[0]) - 1):
        bestAccuracy = 0
        bestFeature = 0
        bestSet = []
        for i in range(len(data[0])):
            featureTemp = cp.copy(featureList)
            if i == 0:
                continue
            if i in featureList:
                continue
            featureTemp.append(i)
            feature = data[:, featureTemp]
            result = 100 * crossValidation(feature, labels)
            if result > bestAccuracy:
                bestAccuracy = result
                bestFeature = i
                bestSet = featureTemp
            print('Using Features : ', i , ' : accuracy = {:0.1f}%'.format(result))
        print('Feature set ', bestSet, ' was best : accuracy = {:0.1f}%'.format(bestAccuracy))
        featureList.append(bestFeature)

#Backward Elimination
def backwardElimination(data, labels):
    defaultRate = 100 * co.Counter(labels).get(max(co.Counter(labels))) / len(labels)
    featureList = np.arange(1, len(data[0]))
    featureList = featureList.tolist()
    feature = data[:, featureList]
    result = 100 * crossValidation(feature, labels)
    print('Using Features :', featureList ,': accuracy = {:0.1f}%'.format(result))
    while len(featureList) != 0:
        bestAccuracy = 0
        bestFeature = 0
        bestSet = []
        for i in featureList:
            featureTemp = cp.copy(featureList)
            featureTemp.remove(i)
            feature = data[:, featureTemp]
            result = 100 * crossValidation(feature, labels)
            if result > bestAccuracy:
                bestAccuracy = result
                bestFeature = i
                bestSet = featureTemp
            print('Eliminating Features : ', i , ' : accuracy = {:0.1f}%'.format(result))
        print('Feature set ', bestSet, ' was best : accuracy = {:0.1f}%'.format(bestAccuracy))
        featureList.remove(bestFeature)

backwardElimination(smallDataSet, smallDataLabels)