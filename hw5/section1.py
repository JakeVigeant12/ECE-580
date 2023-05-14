import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap

mpl.use('TkAgg')  # !IMPORTANT


def genNormalData(mean, cov):
    numObservations = 100
    data = np.random.multivariate_normal(mean, cov, numObservations)
    return data


def makeClassedDf(data0, data1):
    dataF0 = pd.DataFrame(data0, columns=["x", "y"])
    dataF0["class"] = 0

    dataF1 = pd.DataFrame(data1, columns=["x", "y"])
    dataF1["class"] = 1

    plt.scatter(dataF0["x"].values, dataF0["y"].values, color="blue", label="Class 0")
    plt.scatter(dataF1["x"].values, dataF1["y"].values, color="red", label="Class 1")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("data_generated.png")
   # plt.show()

    return pd.concat([dataF1, dataF0], axis=0)


def plotDataPoints(class0, class1):
    plt.scatter(class0[:, 0], class0[:, 1], color="blue", label="Class 0")
    plt.scatter(class1[:, 0], class1[:, 1], color="red", label="Class 1")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.savefig("data_generated.png")
    # plt.show()


def fitDependBayes(data, data0, data1):
    # gen the gaussian for classes 1 and 2
    class0 = pd.DataFrame(data0)
    class1 = pd.DataFrame(data1)
    # generate full and unique covariance
    sharedCov = generateSameCov(data0,data1)
    cov1 = sharedCov
    cov2 = sharedCov
    mean1 = class0.mean(axis=0)
    mean2 = class1.mean(axis=0)
    # P(x1,x2|class) liklihood
    class0Liklihood = multivariate_normal(mean=mean1, cov=cov1)
    class1Liklihood = multivariate_normal(mean=mean2, cov=cov2)

    X = data.iloc[:, 0:2]

    xcoords = X.iloc[:, 0].values
    ycoords = X.iloc[:, 1].values

    x_min, x_max = xcoords.min() - 1, xcoords.max() + 1
    y_min, y_max = ycoords.min() - 1, ycoords.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    points = np.column_stack([xx.ravel(), yy.ravel()])

    pdf1 = class0Liklihood.pdf(points)
    pdf2 = class1Liklihood.pdf(points)
    decision_statistic = np.log(pdf2 / pdf1)
    Z = decision_statistic.reshape(xx.shape)
    levels = np.linspace(Z.min(), Z.max(), 100)
    plotDataPoints(data0, data1)
    plt.contour(xx, yy, Z, levels=[0], colors='black', alpha=0.7)
    plt.contourf(xx, yy, Z, levels=levels, cmap = plt.cm.RdBu_r,alpha=0.7)

    plt.colorbar()
    plt.title("Full Classifier")

    plt.savefig("depend_same_cov_results.png")
    plt.show()


def fitIndepBayesFull(data, class0, class1):
    X = data.iloc[:, 0:2]

    xcoords = X.iloc[:, 0].values
    ycoords = X.iloc[:, 1].values

    x_min, x_max = xcoords.min() - 1, xcoords.max() + 1
    y_min, y_max = ycoords.min() - 1, ycoords.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    X1c0 = class0[:, 0]
    X2c0 = class0[:, 1]
    X1c1 = class1[:, 0]
    X2c1 = class1[:, 1]
    meanX1c0 = np.mean(X1c0)
    meanX2c0 = np.mean(X2c0)
    meanX1c1 = np.mean(X1c1)
    meanX2c1 = np.mean(X2c1)
    cov = generateSameCov(class0, class1)
    varX1c0 = cov[0][0]#
    varX2c0 = cov[1][1]#
    varX1c1 = cov[0][0]#
    varX2c1 = cov[1][1]#
    # make predictions from distributions for each point
    pdf_xc0 = norm.pdf(xx, loc=meanX1c0, scale=np.sqrt(varX1c0))
    pdf_xc1 = norm.pdf(xx, loc=meanX1c1, scale=np.sqrt(varX1c1))
    pdf_yc1 = norm.pdf(yy, loc=meanX2c1, scale=np.sqrt(varX2c1))
    pdf_yc0 = norm.pdf(yy, loc=meanX2c0, scale=np.sqrt(varX2c0))
    likC1 = pdf_xc1 * pdf_yc1
    likC0 = pdf_xc0 * pdf_yc0
    zz = np.log(likC1 / likC0)

    Z = zz.reshape(xx.shape)

    levels = np.linspace(Z.min(), Z.max(), 100)

    plotDataPoints(class0, class1)
    # boundary plot
    plt.contour(xx, yy, Z, levels=[0], colors='black')
    # level curves
    plt.contourf(xx, yy, Z, levels=levels, cmap = plt.cm.RdBu_r, alpha=0.5)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Naive Classifier Boundary')
    plt.savefig("indep_classifier_same")
    plt.show()


def generateSameCov(data0, data1):
    # in as np arrays
    mVec0 = np.mean(data0, axis=0)
    mVec1 = np.mean(data1, axis=0)
    deMean0 = data0 - mVec0
    deMean1 = data1 - mVec1
    groupedData = np.append(deMean0, deMean1, axis=0)
    cov_matrix = np.cov(groupedData, rowvar=False)
    return cov_matrix


# cov = correlCoeff * sigma_x * sigma_y
# selected correlCoeff = 0.5, s_x = 1, s_y = 2
covClass0 = [[1, 1], [1, 4]]
covClass1 = [[1, -1], [-1, 4]]
mean0 = [0, 0]
mean1 = [0, 5]
dataClass0 = genNormalData(mean0, covClass0)
dataClass1 = genNormalData(mean1, covClass1)
data = makeClassedDf(dataClass0, dataClass1)
fitIndepBayesFull(data, dataClass0, dataClass1)
