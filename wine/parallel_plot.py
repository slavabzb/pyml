import pylab
import math
import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/wine-quality/winequality-red.csv")


def main():
    wine = pd.read_csv(TARGET_URL, header=0, sep=';')
    summary = wine.describe()
    nrows = len(wine.index)
    tasteCol = len(summary.columns)
    meanTaste = summary.iloc[1, tasteCol - 1]
    sdTaste = summary.iloc[2, tasteCol - 1]
    nDataCol = len(wine.columns) - 1

    for i in range(nrows):
        dataRow = wine.iloc[i, 1:nDataCol]
        normTarget = (wine.iloc[i, nDataCol] - meanTaste) / sdTaste
        labelColor = 1.0 / (1.0 + math.exp(-normTarget))
        dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attribute Values')
    plt.grid()
    plt.show()

    normalized = wine
    ncols = len(normalized.columns)

    for i in range(ncols):
        mean = summary.iloc[1, i]
        sd = summary.iloc[2, i]
        normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

    for i in range(nrows):
        dataRow = normalized.iloc[i, 1:nDataCol]
        normTarget = normalized.iloc[i, nDataCol]
        labelColor = 1.0 / (1.0 + math.exp(-normTarget))
        dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attribute Values')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
