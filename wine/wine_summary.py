import pylab
import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/wine-quality/winequality-red.csv")


def main():
    wine = pd.read_csv(TARGET_URL, header=0, sep=';')
    print(wine.head())

    summary = wine.describe()
    print(summary)

    normalized = wine
    ncols = len(normalized.columns)

    for i in range(ncols):
        mean = summary.iloc[1, i]
        sd = summary.iloc[2, i]
        normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

    array = normalized.values
    pylab.boxplot(array)

    plt.xlabel('Attribute Index')
    plt.ylabel('Quartile Ranges - Normalized')
    pylab.show()

if __name__ == '__main__':
    main()
