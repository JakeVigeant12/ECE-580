from math import floor

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.stats import gaussian_kde
import seaborn as sns

mpl.use('TkAgg')  # !IMPORTANT


def getData(fileName):
    df = pd.read_csv('./data/' + fileName + ".csv")
    return df


def applyRule(dataFrame):
    trueClass = dataFrame.iloc[:, 0]
    decisionStats = dataFrame.iloc[:, 1]
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(len(decisionStats)):
        #The rule may be incorrect but the code functions properly
        if decisionStats[i] >= 1/3:
            classification = np.random.binomial(n=1, p=0.95)
        else:
            classification = np.random.binomial(n=1, p=0.825)
        if classification == 1 and trueClass[i] == 1:
            tp += 1
        elif classification == 0 and trueClass[i] == 1:
            fn += 1
        elif classification == 0 and trueClass[i] == 0:
            tn += 1
        elif classification == 1 and trueClass[i] == 0:
            fp += 1
    fpr = fp / (fp + tn)
    tdr = tp / (tp + fn)
    return tdr, fpr


def genExpect(currData):
    fprEstimates = []
    tdrEstimates = []
    for i in range(100):
        tdr, fpr = applyRule(currData)
        tdrEstimates.append(tdr)
        fprEstimates.append(fpr)
    sns.kdeplot(fprEstimates)
    plt.grid(True)
    plt.title('Pfa PDF')
    plt.show()
    sns.kdeplot(tdrEstimates)
    plt.grid(True)
    plt.title('Pd PDF')
    plt.show()
    print("Expectation of Pfa " + str(np.mean(np.array(fprEstimates))))
    print("Expectation of Pd " + str(np.mean(np.array(tdrEstimates))))
    return fprEstimates, tdrEstimates


def genAllStatThreshold(data):
    stats = data.iloc[:, 1].values
    stats = np.concatenate((stats, [-np.inf, np.inf]))
    stats = np.sort(stats)
    return stats


def plotROC(dataFrame, fprLis, tdrLis):
    trueClass = dataFrame.iloc[:, 0]
    decisionStats = dataFrame.iloc[:, 1]

    threshies = [genAllStatThreshold(dataFrame)]

    fig, axs = plt.subplots(1, 5, figsize=(30, 30), sharey=True)
    plt.title('ROC Curve ')
    for ax in axs:
        ax.set_aspect('equal')

    i = 0
    for thresholds in threshies:
        fpr = []
        tdr = []
        for threshold in thresholds:
            threshPredict = (decisionStats >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(trueClass, threshPredict).ravel()
            tdr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
        axs[i].plot(fpr, tdr)
        axs[i].set_xlabel('Probability False Alarm')
        axs[i].set_ylabel('Probability Detection')
    axs[0].scatter(fprLis, tdrLis)
    axs[0].scatter(np.mean(fprLis), np.mean(tdrLis))
    plt.show()


currData = getData("knn3DecisionStatistics")
# fprEstimates, tdrEstimates
fpr, tdr = genExpect(currData)
plotROC(currData, fpr, tdr)
