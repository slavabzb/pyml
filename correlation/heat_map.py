import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    corMat = pd.DataFrame(dataset.corr())
    plt.pcolor(corMat)
    plt.show()


if __name__ == '__main__':
    main()
