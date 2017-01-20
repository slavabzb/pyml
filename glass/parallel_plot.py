import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = ("https://archive.ics.uci.edu/ml/machine-"
              "learning-databases/glass/glass.data")


def main():
    glass = pd.read_csv(TARGET_URL, header=None, prefix='V')
    glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si',
                     'K', 'Ca', 'Ba', 'Fe', 'Type']

    normalized = glass
    ncols = len(normalized.columns)
    nrows = len(normalized.index)
    summary = normalized.describe()
    nDataCol = ncols - 1

    for i in range(nDataCol):
        mean = summary.iloc[1, i]
        sd = summary.iloc[2, i]
        normalized.iloc[:, i:(i+1)] = (normalized.iloc[:, i:(i+1)] - mean) / sd

    for i in range(nrows):
        dataRow = normalized.iloc[i, 1:nDataCol]
        labelColor = normalized.iloc[i, nDataCol] / 7.0
        dataRow.plot(color=plt.cm.RdYlBu(labelColor), alpha=0.5)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attrubute Values')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
