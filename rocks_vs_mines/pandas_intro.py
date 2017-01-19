import pandas as pd

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    print(dataset.head())
    print(dataset.tail())

    summary = dataset.describe()
    print(summary)


if __name__ == '__main__':
    main()
