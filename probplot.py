import pylab
import scipy.stats as stats
import urllib2


TARGET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-'\
             'databases/undocumented/connectionist-bench/sonar/sonar.all-data'


def main():
    data = urllib2.urlopen(TARGET_URL)
    xList = [line.strip().split(',') for line in data]

    col = 3
    colData = [float(row[col]) for row in xList]

    stats.probplot(colData, dist='norm', plot=pylab)
    pylab.show()


if __name__ == '__main__':
    main()
