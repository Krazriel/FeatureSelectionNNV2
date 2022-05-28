import numpy as np
import collections as co
import copy as cp

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
    level = 0
    print('[ LEVEL' , level ,'] : Initial Features :  []  : accuracy = {:0.1f}%'.format(defaultRate))
    featureList = []

    while(len(featureList) != len(data[0]) - 1):
        level += 1
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
        print('[ LEVEL' , level ,'] Feature set ', bestSet, ' was best : accuracy = {:0.1f}%'.format(bestAccuracy))
        featureList.append(bestFeature)

#Backward Elimination
def backwardElimination(data, labels):
    level = 0
    defaultRate = 100 * co.Counter(labels).get(max(co.Counter(labels))) / len(labels)
    featureList = np.arange(1, len(data[0]))
    featureList = featureList.tolist()
    feature = data[:, featureList]
    result = 100 * crossValidation(feature, labels)
    print('[ LEVEL' , level ,'] : Initial Features :', featureList ,': accuracy = {:0.1f}%'.format(result))
    while len(featureList) != 0:
        level += 1
        bestAccuracy = 0
        bestFeature = 0
        bestSet = []
        for i in featureList:
            featureTemp = cp.copy(featureList)
            featureTemp.remove(i)
            if len(featureTemp) == 0:
                print('Eliminating Features : ', i  ,': accuracy = {:0.1f}%'.format(defaultRate))
                print('[ LEVEL' , level ,'] : Feature set ', bestSet, ' was best : accuracy = {:0.1f}%'.format(defaultRate))
                return
            feature = data[:, featureTemp]
            result = 100 * crossValidation(feature, labels)
            if result > bestAccuracy:
                bestAccuracy = result
                bestFeature = i
                bestSet = featureTemp
            print('Eliminating Features : ', i , ' : accuracy = {:0.1f}%'.format(result))
        print('[ LEVEL' , level ,'] : Feature set ', bestSet, ' was best : accuracy = {:0.1f}%'.format(bestAccuracy))
        featureList.remove(bestFeature)
    

def main():
    userChoice = input('Enter small, large, custom: ')
    if userChoice == 'small':
        smallDataSet = np.loadtxt('data/CS205_SP_2022_SMALLtestdata__38.txt')
        smallDataLabels = smallDataSet[:, 0]
        algorithmChoice = input('Enter forward, backward: ')

        if algorithmChoice == 'forward':
            forwardSelection(smallDataSet, smallDataLabels)
        elif algorithmChoice == 'backward':
            backwardElimination(smallDataSet, smallDataLabels)
        else:
            print('Invalid Input: must be forward or backward')

    elif userChoice == 'large':
        largeDataSet = np.loadtxt('data/CS205_SP_2022_Largetestdata__3.txt')
        largeDataLabels = largeDataSet[:, 0]
        algorithmChoice = input('Enter forward, backward: ')

        if algorithmChoice == 'forward':
            forwardSelection(largeDataSet, largeDataLabels)
        elif algorithmChoice == 'backward':
            backwardElimination(largeDataSet, largeDataLabels)
        else:
            print('Invalid Input: must be forward or backward')

    elif userChoice == 'custom':
        customDataSet = np.loadtxt('data/bodyPerformance_new.txt', delimiter=',')
        customDataLabel = customDataSet[:, 0]
        algorithmChoice = input('Enter forward, backward: ')

        if algorithmChoice == 'forward':
            forwardSelection(customDataSet, customDataLabel)
        elif algorithmChoice == 'backward':
            backwardElimination(customDataSet, customDataLabel)
        else:
            print('Invalid Input: must be forward or backward')

    else:
        print('Invalid Input: must be small, large, or custom')


if __name__ == "__main__":
    main()