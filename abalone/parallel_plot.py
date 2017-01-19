import pandas as pd
import math
import matplotlib.pyplot as plt

TARGET_URL = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")


def main():
    abalone = pd.read_csv(TARGET_URL, header=None, prefix='V')
    abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
                       'Whole weight', 'Shucked weight', 'Viscera weight',
                       'Shell weight', 'Rings']

    summary = abalone.describe()
    minRings = summary.iloc[3, 7]
    maxRings = summary.iloc[7, 7]
    nrows = len(abalone.index)

    for i in range(nrows):
        dataRow = abalone.iloc[i, 1:8]
        labelColor = (abalone.iloc[i, 8] - minRings) / (maxRings - minRings)
        dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attribute Values')
    plt.show()

    meanRings = summary.iloc[1, 7]
    sdRings = summary.iloc[2, 7]

    for i in range(nrows):
        dataRow = abalone.iloc[i, 1:8]
        normTarget = (abalone.iloc[i, 8] - meanRings) / sdRings
        labelColor = 1.0 / (1.0 + math.exp(-normTarget))
        dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attribute Values')
    plt.show()

if __name__ == '__main__':
    main()
