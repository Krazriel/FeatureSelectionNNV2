import numpy as np
import collections as co

#Set up data, features and labels sets
smallDataSet = np.loadtxt('data/CS205_SP_2022_SMALLtestdata__38.txt')
largeDataSet = np.loadtxt('data/CS205_SP_2022_Largetestdata__3.txt')

smallDataLabels = smallDataSet[:, 0]
largeDataLabels = largeDataSet[:, 0]

smallData = smallDataSet[:, 1:]
largeData = largeDataSet[:, 1:]


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

#ForwardSelection
def forwardSelection(data, labels):
    defaultRate = 100 * co.Counter(labels).get(max(co.Counter(labels))) / len(labels)
    featureContainer = []
    print('Using Features : ', featureContainer, ' : accuracy = {:0.1f}%'.format(defaultRate))

def backwardElimination(data, labels):
    defaultRate = 100 * co.Counter(labels).get(max(co.Counter(labels))) / len(labels)
