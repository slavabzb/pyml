import pandas as pd
import pylab
import matplotlib.pyplot as plt

TARGET_URL = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")


def main():
    abalone = pd.read_csv(TARGET_URL, header=None, prefix='V')
    abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
                       'Whole weight','Shucked weight', 'Viscera weight',
                       'Shell weight', 'Rings']

    print(abalone.head())
    print(abalone.tail())

    summary = abalone.describe()
    print(summary)

    array = abalone.iloc[:, 1:9].values
    pylab.boxplot(array)
    plt.xlabel('Attribute Index')
    plt.ylabel('Quartile Ranges')
    pylab.show()

    array = abalone.iloc[:, 1:8].values
    pylab.boxplot(array)
    plt.xlabel('Attribute Index')
    plt.ylabel('Quartile Ranges')
    pylab.show()

    normalized = abalone.iloc[:, 1:9]

    for i in range(8):
        mean = summary.iloc[1, i]
        sd = summary.iloc[2, i]
        normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

    array = normalized.values
    pylab.boxplot(array)
    plt.xlabel('Attribute Index')
    plt.ylabel('Quartile Ranges')
    pylab.show()

if __name__ == '__main__':
    main()
