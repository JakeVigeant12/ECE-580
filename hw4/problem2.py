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
    dataF = pd.read_csv(FILE_PATHS + name + ".csv")
    labels = dataF.iloc[:0].values
    dataVecs = dataF.drop(dataF.columns[0], axis = 1).values
    return dataF


def splitByFoldAssignment(df):
    df.columns = ['FoldAssignment', 'TrueClass', 'FeatureVectorX', 'FeatureVectorY']
    grouped = df.groupby('FoldAssignment')
    fold1 = grouped.get_group(2)
    fold2 = grouped.get_group(1)
    fold1 = fold1.drop('FoldAssignment', axis=1)
    fold2 = fold2.drop('FoldAssignment', axis=1)
    df = df.drop('FoldAssignment', axis=1)
    return fold1, fold2, df


def plotROC(dataFrame, title):
    # true class stored in first column
    trueClass = dataFrame.iloc[:, 0]
    # compute desicion stats and store in col 2
    decisionStats = dataFrame.iloc[:, 1]
    # fixed for k=5 classifier
    thresholds = [0, 0.2, 0.4, 0.6, 0.8, 1]
    tdr = []
    fpr = []
    for threshold in thresholds:
        threshPredict = (decisionStats >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(trueClass, threshPredict).ravel()
        tdr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    plt.xlabel('Probability False Alarm')
    plt.ylabel('Probability Detection')
    plt.grid()
    plt.title(title)
    plt.plot(fpr, tdr)
    plt.savefig(title + ".png")
    plt.show()


def distance(samp1, samp2):
    dist = 0
    for i in range(len(samp1)):
        dist += (samp1[i] - samp2[i]) ** 2
    return dist ** 0.5


def applyKNN(fullDf, dFold1, dFold2):
    # test on both folds, then take lambda from when each point was tested
    lambdatest2 = genDecisionStatistics(dFold1, dFold2)
    lambdatest1 = genDecisionStatistics(dFold2, dFold1)
    labels1 = dFold1['TrueClass'].values
    labels2 = dFold2['TrueClass'].values
    labels = np.concatenate((labels1, labels2))
    lambdas = np.concatenate((lambdatest1, lambdatest2))
    dfCV = pd.DataFrame({'class': labels, 'lambda': lambdas})



    plotROC(dfCV, "New CV")

def plotDecisionSurface(dataVectors, labels,title):
    x_min, x_max = dataVectors[:, 0].min() - 1, dataVectors[:, 0].max() + 1
    y_min, y_max = dataVectors[:, 1].min() - 1, dataVectors[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = np.array(genPredictions(np.c_[xx.ravel(), yy.ravel()],dataVectors,labels))
    Z = Z.reshape(xx.shape)
    # Plot the decision surface
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(dataVectors[:, 0], dataVectors[:, 1], c=labels, cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.savefig(title+".png")

    plt.show()

def performRandomCV(df):
    # Assuming your dataframe is called 'df' and the target column is called 'target'
    X = df.drop('TrueClass', axis=1)
    y = df['TrueClass']

    # Initialize the StratifiedShuffleSplit function with 2 splits
    ss =  StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Create a new

    # Create a new column in the dataframe to store the fold number
    df['FoldAssignment'] = np.nan
    # Loop through the splits
    for fold_idx, (train_index, test_index) in enumerate(ss.split(X, y)):
        # Get the training and test sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Assign fold number to the corresponding rows in the dataframe
        df.loc[test_index, 'FoldAssignment'] = fold_idx+1
    return df


def genDecisionStatistics(trainData, testData):
    lambdaList = []
    testDataVectors = testData.drop('TrueClass', axis=1)
    testDataVectors = testDataVectors.values
    testLabels = testData['TrueClass'].values

    trainingVectors = trainData.drop('TrueClass', axis=1)
    trainingVectors = trainingVectors.values
    trainingLabels = trainData['TrueClass'].values

    for testPoint in testDataVectors:
        # Do this for all testing points
        distances = []
        for i, trainingPoint in enumerate(trainingVectors):
            distanceS = distance(trainingPoint, testPoint)
            distances.append((i, distanceS))
        distances = sorted(distances, key=lambda x: x[1])
        neighbors = []
        for i in range(5):
            index = distances[i][0]
            # get the neighboring vector and its class
            neighbor = (trainingVectors[index], trainingLabels[index])
            neighbors.append(neighbor)
        # poll the neighbors to get lambda (counted numbers of H1)
        lamda = 0
        for neighbor in neighbors:
            if (neighbor[1] == 1):
                lamda += 1
        lamda /= 5
        lambdaList.append(lamda)
    return lambdaList

def genPredictions(dataPoints, classifierData, classifierLabels):
    predictionList = []
    for dataPoint in dataPoints:
        distances = []
        for i, trainingPoint in enumerate(classifierData):
            distanceS = distance(trainingPoint, dataPoint)
            distances.append((i, distanceS))
        distances = sorted(distances, key=lambda x: x[1])
        neighbors = []
        for i in range(5):
            index = distances[i][0]
            # get the neighboring vector and its class
            neighbor = (classifierData[index], classifierLabels[index])
            neighbors.append(neighbor)
            # poll the neighbors to get lambda (counted numbers of H1)
        lamda = 0
        for neighbor in neighbors:
            if (neighbor[1] == 1):
                lamda += 1
        lamda /= 5
        if (lamda > 0.4):
            predictionList.append(1)
        else:
            predictionList.append(0)
    return predictionList

data = loadData("CrossValWithKeys")
fold1, fold2, incData = splitByFoldAssignment(data)
betterCv = performRandomCV(data)
realFold1, realFold2, realInces = splitByFoldAssignment(betterCv)
applyKNN(realFold1, realFold2, realInces)
# generate a df with decision statistics
print("hello")