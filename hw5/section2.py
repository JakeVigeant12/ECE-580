import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import matplotlib.colors as mcolors

mpl.use('TkAgg')  # !IMPORTANT

DATA_PATH = "./dataSet"


def loadData(num):
    df = pd.read_csv(DATA_PATH + str(num) + ".csv")
    return df


def logistic(data):
    labels = data.iloc[:, 0].values
    vectors = data.drop(data.columns[0], axis=1).values
    logis = LogisticRegression()
    logis.fit(vectors, labels)
    # Meshgrid to find lambda(x) across
    x_min, x_max = vectors[:, 0].min() - 1, vectors[:, 0].max() + 1
    y_min, y_max = vectors[:, 1].min() - 1, vectors[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = logis.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    levels = np.linspace(Z.min(), Z.max(), 100)
    plt.contour(xx, yy, Z, levels=[0.5], colors='orange', alpha=0.7)
    plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.RdBu_r, alpha=0.5)



def linearDiscr(data):
    labels = data.iloc[:, 0].values
    vectors = data.drop(data.columns[0], axis=1).values
    lda = LinearDiscriminantAnalysis()
    lda.fit(vectors, labels)
    # Meshgrid to find lambda(x) across
    x_min, x_max = vectors[:, 0].min() - 1, vectors[:, 0].max() + 1
    y_min, y_max = vectors[:, 1].min() - 1, vectors[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    # Plot the decision boundary where lambda(x) = 0
    levels = np.linspace(Z.min(), Z.max(), 100)
    plt.contour(xx, yy, Z, levels=[0.5], colors='red', alpha=0.7)
    plt.contourf(xx, yy, Z, levels=levels, cmap= plt.cm.RdBu_r, alpha=0.5)



def plotDataPoints(data):
    class_0 = data[data.iloc[:, 0] == 0]
    class_1 = data[data.iloc[:, 0] == 1]
    class_0 = class_0.iloc[:, 1:].values
    class_1 = class_1.iloc[:, 1:].values
    # plot the data with different colors for each class
    plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
    plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')


def fitDependBayes(data, data0, data1):
    # gen the gaussian for classes 1 and 2
    class0 = pd.DataFrame(data0)
    class1 = pd.DataFrame(data1)
    # generate full and unique covariance
    cov1 = class0.cov()
    cov2 = class1.cov()
    mean1 = class0.mean(axis=0)
    mean2 = class1.mean(axis=0)
    # P(x1,x2|class) liklihood
    class0Liklihood = multivariate_normal(mean=mean1, cov=cov1)
    class1Liklihood = multivariate_normal(mean=mean2, cov=cov2)

    X = data.iloc[:, 1:]

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

    plt.contour(xx, yy, Z, levels=[0], colors='black', alpha=0.7)
    plt.contourf(xx, yy, Z, levels=levels, cmap= plt.cm.RdBu_r, alpha=0.7)

    plt.colorbar()
    plt.title("All Classifiers Superimposed")
    plt.savefig("all.png")
    plt.show()



df = loadData(4)
class_0 = df[df.iloc[:, 0] == 0].iloc[:, 1:]
class_1 = df[df.iloc[:, 0] == 1].iloc[:, 1:]
plotDataPoints(df)
linearDiscr(df)
logistic(df)
fitDependBayes(df, class_0, class_1)
