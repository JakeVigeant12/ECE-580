import math

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

mpl.use('TkAgg')  # !IMPORTANT
FILE_PATHS = "./data/dataSet"


def loadData(name):
    dataF = pd.read_csv(FILE_PATHS + name + ".csv", header=None)
    dataTest = pd.read_csv(FILE_PATHS + "HorseshoesTest" + ".csv", header=None)
    labels = dataF.iloc[:, 0].values
    dataVecs = dataF.drop(dataF.columns[0], axis=1).values
    plotDecisionSurface(dataVecs, labels, 399, "k=" + str(399))

    peList = []
    kList = np.arange(1, 399, 15)
    for k in kList:
        lambdas = genDecisionStatistics(dataF, dataF, k)
        rocDf = pd.DataFrame({'class': labels, 'lambda': lambdas})
        peCurr = plotROC(rocDf, "ROC(test)k=" + str(k), k)
        peList.append(peCurr)
    kList = np.divide(kList, dataF.shape[0])
    plt.plot(kList, peList)
    plt.xlabel("K/N")
    plt.ylabel("Pe")
    plt.title("CrossValidated")
    plt.savefig("crossval.png")
    plt.show()


def plotROC(dataFrame, title, k):
    # true class stored in first column
    trueClass = dataFrame.iloc[:, 0]
    # compute desicion stats and store in col 2
    decisionStats = dataFrame.iloc[:, 1]
    thresholds = []
    for i in range(k):
        thresholds.append(i / k)
    thresholds.append(1)

    tdr = []
    fpr = []
    pcdLis = []
    for threshold in thresholds:
        threshPredict = (decisionStats >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(trueClass, threshPredict).ravel()
        pcd = (tn + tp) / (tn + fp + fn + tp)
        pcdLis.append(pcd)
        tdr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    pemin = 1 - (max(pcdLis))
    # plt.xlabel('Probability False Alarm')
    # plt.ylabel('Probability Detection')
    # plt.grid()
    # plt.title(title)
    # plt.plot(fpr, tdr)
    # plt.savefig(title + ".png")
    # plt.show()
    return pemin


def genDecisionStatistics(trainData, testData, k):
    lambdaList = []
    testDataVectors = testData.drop(testData.columns[0], axis=1)
    testDataVectors = testDataVectors.values
    testLabels = testData.iloc[:, 0].values

    trainingVectors = trainData.drop(trainData.columns[0], axis=1)
    trainingVectors = trainingVectors.values
    trainingLabels = trainData.iloc[:, 0].values

    for testPoint in testDataVectors:
        # Do this for all testing points
        distances = []
        for i, trainingPoint in enumerate(trainingVectors):
            distanceS = distance(trainingPoint, testPoint)
            distances.append((i, distanceS))
        distances = sorted(distances, key=lambda x: x[1])
        neighbors = []
        for i in range(k):
            index = distances[i][0]
            # get the neighboring vector and its class
            neighbor = (trainingVectors[index], trainingLabels[index])
            neighbors.append(neighbor)
        # poll the neighbors to get lambda (counted numbers of H1)
        lamda = 0
        for neighbor in neighbors:
            if (neighbor[1] == 1):
                lamda += 1
        lamda /= k
        lambdaList.append(lamda)
    return lambdaList


def plotDecisionSurface(dataVectors, labels, k, title):
    x_min, x_max = dataVectors[:, 0].min() - 1, dataVectors[:, 0].max() + 1
    y_min, y_max = dataVectors[:, 1].min() - 1, dataVectors[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = np.array(genPredictions(np.c_[xx.ravel(), yy.ravel()], dataVectors, labels, k))
    Z = Z.reshape(xx.shape)
    # Plot the decision surface
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    levels = np.arange(1, 399, 1)
    levels = levels / 399



    plt.contour(xx, yy, Z, levels=[0.49875], linewidths=2, colors='k')
    plt.scatter(dataVectors[:, 0], dataVectors[:, 1], c=labels, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.savefig(title + ".png")

    plt.show()


def genPredictions(dataPoints, classifierData, classifierLabels, k):
    predictionList = []
    for dataPoint in dataPoints:
        distances = []
        for i, trainingPoint in enumerate(classifierData):
            distanceS = distance(trainingPoint, dataPoint)
            distances.append((i, distanceS))
        distances = sorted(distances, key=lambda x: x[1])
        neighbors = []
        for i in range(k):
            index = distances[i][0]
            # get the neighboring vector and its class
            neighbor = (classifierData[index], classifierLabels[index])
            neighbors.append(neighbor)
            # poll the neighbors to get lambda (counted numbers of H1)
        lamda = 0
        for neighbor in neighbors:
            if (neighbor[1] == 1):
                lamda += 1
        lamda /= k
        predictionList.append(lamda)
    return predictionList


def distance(samp1, samp2):
    dist = 0
    for i in range(len(samp1)):
        dist += (samp1[i] - samp2[i]) ** 2
    return dist ** 0.5


data = loadData("Horseshoes")
