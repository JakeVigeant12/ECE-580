import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from skrvm import RVC
import os.path
from joblib import dump, load


mpl.use('TkAgg')  # !IMPORTANT

DATA_PATH = "./data/dataSet"


def loadData(name):
    df = pd.read_csv(DATA_PATH + name + ".csv", header=None)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    return X.values, y.values


def applyLinearSvm(X, y):
    # use a linear kernel
    model = svm.SVC(kernel='linear')
    # fit to data
    model.fit(X, y)
    plotDecisionSurface(X, y, model, "Linear SVM")


def applyLinearRvm(X, y):
    #cache model as it was taking a while to run whilst debugging
    if os.path.isfile('rvc_linear.joblib'):
        # Load the trained model from cache
        model = load('rvc_linear.joblib')

    else:
        model = RVC(kernel="linear")
        model.fit(X, y)
        dump(model, 'rvc_linear.joblib')

    plotRVMDecisionSurface(X, y, model, "Linear RVM")


def applyRBFSvm(X, y):
    model = svm.SVC(kernel="rbf")
    model.fit(X, y)
    plotDecisionSurface(X, y, model, "RBF SVM")


def applyRBFRvm(X, y):
    #cache model as it was taking a while to run whilst debugging
    if os.path.isfile('rvc_rbf.joblib'):
        # Load the trained model from cache
        model = load('rvc_rbf.joblib')

    else:
        model = RVC(kernel="rbf", gamma=1)
        model.fit(X, y)
        dump(model, 'rvc_rbf.joblib')

    plotRVMDecisionSurface(X, y, model, "RBF RVM")


def plotDecisionSurface(X, y, model, modelTitle):
    # Produce meshgrid to visualize
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    # Calculate the decision function of the SVM on the meshgrid points
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the decision statistic surface and support vectors as line vectors with colored edges
    levelsTot = np.arange(Z.min(), Z.max(), 0.1)
    plt.contourf(xx, yy, Z, levels=levelsTot, cmap=plt.cm.PuBu, alpha=0.5)
    # boundary
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBu)
    plt.colorbar()

    # plot support vectors as symbols overlaid on points
    support_vectors = model.support_vectors_
    support_vectors_indices = model.support_
    support_vector_classes = y[support_vectors_indices]
    print(support_vector_classes)
    markers = ['o', '^']
    labels = ['Class 0', 'Class 1']
    for i, marker, label in zip(range(len(markers)), markers, labels):
        plt.scatter(support_vectors[support_vector_classes == i, 0],
                    support_vectors[support_vector_classes == i, 1],
                    marker=marker, s=80, facecolors='none', edgecolors='k', label=label)

    plt.title(modelTitle)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig(modelTitle+".png")
    plt.show()


def plotRVMDecisionSurface(X, y, model, modelTitle):
    # Create a grid of points for the decision statistic surface
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape, order='F')

    # Plot the decision statistic surface and support vectors
    levels = np.arange(0, 1, 0.001)
    plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.RdBu, alpha=0.5)
    plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu)
    plt.colorbar()

    # plot support vectors as symbols overlaid on points
    support_vectors = model.relevance_
    support_vector_classes = []
    for vec in support_vectors:
        support_vector_classes.append(model.predict(vec.reshape(1, -1)))
    support_vector_classes = np.array(support_vector_classes)
    support_vector_classes = np.concatenate(support_vector_classes)

    print(support_vector_classes)
    markers = ['o', '^']
    labels = ['Class 0', 'Class 1']
    for i, marker, label in zip(range(len(markers)), markers, labels):
        plt.scatter(support_vectors[support_vector_classes == i, 0],
                    support_vectors[support_vector_classes == i, 1],
                    marker=marker, s=80, facecolors='none', edgecolors='k', label=label)

    plt.legend()
    plt.title(modelTitle)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.savefig(modelTitle+".png")
    plt.show()


X, y = loadData("Horseshoes")
applyRBFRvm(X, y)
