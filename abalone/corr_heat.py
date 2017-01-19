import pandas as pd
import matplotlib.pyplot as plt

TARGET_URL = ("http://archive.ics.uci.edu/ml/machine-"
              "learning-databases/abalone/abalone.data")


def main():
    abalone = pd.read_csv(TARGET_URL, header=None, prefix='V')
    abalone.columns = ['Sex', 'Length', 'Diameter', 'Height',
                       'Whole weight', 'Shucked weight', 'Viscera weight',
                       'Shell weight', 'Rings']

    corMat = pd.DataFrame(abalone.iloc[:, 1:9].corr())
    print(corMat)

    plt.pcolor(corMat)
    plt.show()

if __name__ == '__main__':
    main()
