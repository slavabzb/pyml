import pandas as pd
import math

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    dataset = pd.read_csv(TARGET_URL, header=None, prefix='V')

    datarow2 = dataset.iloc[1, 0:60]
    datarow3 = dataset.iloc[2, 0:60]
    datarow21 = dataset.iloc[20, 0:60]

    mean2 = 0; mean3 = 0; mean21 = 0
    numElt = len(datarow2)
    for i in range(numElt):
        mean2 += datarow2[i] / numElt
        mean3 += datarow3[i] / numElt
        mean21 += datarow21[i] / numElt

    var2 = 0; var3 = 0; var21 = 0
    for i in range(numElt):
        var2 += (datarow2[i] - mean2) * (datarow2[i] - mean2) / numElt
        var3 += (datarow3[i] - mean3) * (datarow3[i] - mean3) / numElt
        var21 += (datarow21[i] - mean21) * (datarow21[i] - mean21) / numElt

    corr23 = 0; corr221 = 0
    for i in range(numElt):
        corr23 += (datarow2[i] - mean2) * (datarow3[i] - mean3) / (math.sqrt(var2 * var3) * numElt)
        corr221 += (datarow2[i] - mean2) * (datarow21[i] - mean21) / (math.sqrt(var2 * var21) * numElt)

    print('Correlation between attribute 2 and 3 = {}'.format(corr23))
    print('Correlation between attribute 2 and 21 = {}'.format(corr221))

if __name__ == '__main__':
    main()
