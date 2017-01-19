import pandas as pd
import matplotlib.pyplot as plt

from random import uniform

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    target = []
    for i in range(208):
        if dataset.iat[i, 60] == 'M':
            target.append(1.0)

        else:
            target.append(0.0)

    datacol35 = dataset.iloc[0:208, 35]

    plt.scatter(datacol35, target)
    plt.xlabel('Attribute Value')
    plt.ylabel('Target Value')
    plt.show()

    target = []
    for i in range(208):
        if dataset.iat[i, 60] == 'M':
            target.append(1.0 + uniform(-0.1, 0.1))

        else:
            target.append(0.0 + uniform(-0.1, 0.1))

    plt.scatter(datacol35, target, alpha=0.5, s=120)
    plt.xlabel('Attribute Value')
    plt.ylabel('Target Value')
    plt.show()

if __name__ == '__main__':
    main()
