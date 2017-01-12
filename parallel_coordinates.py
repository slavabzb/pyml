import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    for i in range(208):
        if dataset.iat[i, 60] == 'M':
            pcolor = 'red'

        else:
            pcolor = 'blue'

        datarow = dataset.iloc[i, 0:60]
        datarow.plot(color=pcolor)

    plt.xlabel('Attribute Index')
    plt.ylabel('Attribute Values')
    plt.show()

if __name__ == '__main__':
    main()
