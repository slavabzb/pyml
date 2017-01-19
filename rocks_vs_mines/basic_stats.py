import urllib2
import numpy as np

TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def get_percent_boundaries(array, ntiles):
    return [np.percentile(array, i * 100 / ntiles) for i in range(ntiles + 1)]


def main():
    data = urllib2.urlopen(TARGET_URL)
    xList = [line.strip().split(',') for line in data]

    col = 3
    colData = [float(row[col]) for row in xList]

    colArray = np.array(colData)
    colMean = np.mean(colArray)
    colSd = np.std(colArray)

    print('Mean = {}'.format(colMean))
    print('Standard deviation = {}'.format(colSd))
    print('Boundaries for  4 equal percentiles = {}'.format(get_percent_boundaries(colArray, ntiles=4)))
    print('Boundaries for 10 equal percentiles = {}'.format(get_percent_boundaries(colArray, ntiles=10)))

    col = 60
    colData = [row[col] for row in xList]

    unique = set(colData)
    print('Unique label values = {}'.format(unique))

    catDict = dict(zip(list(unique), range(len(unique))))

    catCount = [0] * 2
    for elt in colData:
        catCount[catDict[elt]] += 1

    print('Counts for each value of categorical label = {}'.format(list(catCount)))


if __name__ == '__main__':
    main()
