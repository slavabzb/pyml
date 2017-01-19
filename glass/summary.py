import pylab
import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = ("https://archive.ics.uci.edu/ml/machine-"
              "learning-databases/glass/glass.data")


def main():
    glass = pd.read_csv(TARGET_URL, header=None, prefix='V')
    glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si',
                     'K', 'Ca', 'Ba', 'Fe', 'Type']

    print(glass.head())

    summary = glass.describe()
    print(summary)

    ncol1 = len(glass.columns)
    normalized = glass.iloc[:, 1:ncol1]

    ncol2 = len(normalized.columns)
    summary2 = normalized.describe()

    for i in range(ncol2):
        mean = summary2.iloc[1, i]
        sd = summary2.iloc[2, i]
        normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

    array = normalized.values
    pylab.boxplot(array)

    plt.xlabel('Attribute Index')
    plt.ylabel('Quartile Ranges - Normalized')
    pylab.show()

if __name__ == '__main__':
    main()
