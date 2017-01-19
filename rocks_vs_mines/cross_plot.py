import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    datarow2 = dataset.iloc[1, 0:60]
    datarow3 = dataset.iloc[2, 0:60]

    plt.scatter(datarow2, datarow3)
    plt.xlabel('2nd Attribute')
    plt.ylabel('3rd Attribute')
    plt.show()

    datarow21 = dataset.iloc[20, 0:60]

    plt.scatter(datarow2, datarow21)

    plt.xlabel('2nd Attribute')
    plt.ylabel('21st Attribute')
    plt.show()

if __name__ == '__main__':
    main()
