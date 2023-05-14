import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from sklearn.model_selection import KFold

dataFile = open('imports-85.txt', 'r')
Lines = dataFile.readlines()
count = 0
for i in range(len(Lines)):
    count += 1
    Lines[i] = Lines[i].strip().split(",")
# Remove extraneous data features, 205 initial points
labels = ["Wheel Base", "Length", "Width", "Height", "Curb Weight", "Engine Size", "Bore", "Stroke",
          "Compression Ratio", "Horsepower", "Peak RPM", "City MPG", "Highway MPG", "Price"]
data = pd.DataFrame(Lines)
data = data.drop(data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 17]], axis=1)
# data.to_csv("imports-85.csv")
auto_data = pd.read_csv('imports-85.csv', names=labels, header=None)
print(auto_data.shape[0])
# Remove data points missing a price, 4 points
auto_data.drop(auto_data[auto_data['Price'] == '?'].index, inplace=True)
print(auto_data.shape[0])
# Remove points missing at least 1 of the 13 remaining attributes, 6 points
for label in labels:
    auto_data.drop(auto_data[auto_data[label] == '?'].index, inplace=True)
print(auto_data.shape[0])
auto_data = auto_data.astype('float64')


# In total, 10 points are removed, leaving 195
# 40 iterations of 5-fold cross valitdation
# There are 5 models calculated for each iteration, 40 iterations = 200 models
def splitData(data):
    # rearrange data rows
    df = data.sample(frac=1)
    X = df.iloc[:, :13].values
    Y = df.iloc[:, 13:].values
    xFolds = []
    yFolds = []
    for i in range(5):
        startI = i * 39
        endI = ((i + 1) * 39)
        xFolds.append(X[startI: endI])
        yFolds.append(Y[startI:endI])
    return xFolds, yFolds


models = []
for i in range(40):
    xFolds, yFolds = splitData(auto_data)
    for i in range(5):
        testX = xFolds[i]
        testY = yFolds[i]
        trainX = []
        trainY = []
        for j in range(5):
            if (j != i):
                trainX.append(xFolds[i])
                trainY.append(yFolds[i])
        trainX = np.concatenate(trainX)
        print("s")
        # pull out the training set.
        # train the model then predict on all the points, including the training points.
        # what do the training-fold y values matter for?
        # model = sm.OLS.fit(splitData[i])
        # models.append(model)

x = auto_data[['Width', 'Bore', 'Horsepower']]
y = auto_data['Price']
model = sm.OLS(y, x).fit()
print_model = model.summary()
print(print_model)
