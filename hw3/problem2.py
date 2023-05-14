from math import floor

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

mpl.use('TkAgg')  # !IMPORTANT


def getData(fileName):
    df = pd.read_csv('./data/' + fileName + ".csv")
    return df



currData = getData("knn3DecisionStatistics")
