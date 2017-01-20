import urllib2
import numpy as np

import pylab as pl

from sklearn import linear_model
from sklearn.metrics import roc_curve, auc

TARGET_URL = ('https://archive.ics.uci.edu/ml/machine-learning-'
              'databases/undocumented/connectionist-bench/sonar/sonar.all-data')


def confusionMatrix(predicted, actual, threshold):
    if len(predicted) != len(actual): return -1

    tp = 0.0; fp = 0.0; tn = 0.0; fn = 0.0

    for i in range(len(actual)):
        if actual[i] > 0.5:
            if predicted[i] > threshold:
                tp += 1.0

            else:
                fn += 1.0

        else:
            if predicted[i] < threshold:
                tn += 1.0

            else:
                fp += 1.0

    return tp, fn, fp, tn

def main():
    data = urllib2.urlopen(TARGET_URL)

    xList = []
    labels = []
    for line in data:
        row = line.strip().split(',')
        labels.append(1.0 if row[-1] == 'M' else 0.0)
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

    indices = range(len(xList))
    xListTest = [xList[i] for i in indices if i % 3 == 0]
    xListTrain = [xList[i] for i in indices if i % 3 != 0]
    labelsTest = [labels[i] for i in indices if i % 3 == 0]
    labelsTrain = [labels[i] for i in indices if i % 3 != 0]

    xTrain = np.array(xListTrain)
    yTrain = np.array(labelsTrain)
    xTest = np.array(xListTest)
    yTest = np.array(labelsTest)

    print('Shape of xTrain array is {}'.format(xTrain.shape))
    print('Shape of yTrain array is {}'.format(yTrain.shape))
    print('Shape of xTest array is {}'.format(xTest.shape))
    print('Shape of yTest array is {}'.format(yTest.shape))

    model = linear_model.LinearRegression()
    model.fit(xTrain, yTrain)

    predicts = model.predict(xTrain)
    print('Some values predicted by model: {}'.format(predicts[0:5]))

    tp, fn, fp, tn = confusionMatrix(predicts, yTrain, 0.5)
    print('tp == {}\tfn = {}\tfp = {}\ttn = {}'.format(tp, fn, fp, tn))

    predictsTest = model.predict(xTest)
    tp, fn, fp, tn = confusionMatrix(predictsTest, yTest, 0.5)
    print('tp == {}\tfn = {}\tfp = {}\ttn = {}'.format(tp, fn, fp, tn))

    fpr, tpr, thresholds = roc_curve(yTrain, predicts)
    roc_auc = auc(fpr, tpr)
    print('AUC for in-sample ROC curve: {}'.format(roc_auc))

    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k-')
    pl.xlim([0, 1])
    pl.ylim([0, 1])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('In-sample ROC rocks vs mines')
    pl.legend(loc='lower right')
    pl.show()

    fpr, tpr, thresholds = roc_curve(yTest, predictsTest)
    roc_auc = auc(fpr, tpr)
    print('AUC for out-of-sample ROC curve: {}'.format(roc_auc))

    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k-')
    pl.xlim([0, 1])
    pl.ylim([0, 1])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Out-of-sample ROC rocks vs mines')
    pl.legend(loc='lower right')
    pl.show()

if __name__ == '__main__':
    main()

