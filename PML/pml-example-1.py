import pandas
from matplotlib import pyplot
import numpy
from perceptron import Perceptron
df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
tail = df.tail()
y = df.iloc[0:100, 4].values
y = numpy.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
# pyplot.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
# pyplot.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# pyplot.xlabel('sepal length [cm]')
# pyplot.ylabel('petal length [cm]')
# pyplot.legend(loc='upper left')
# pyplot.tight_layout()
# pyplot.show()

ppn = Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

pyplot.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
pyplot.xlabel('Epochs')
pyplot.ylabel('Number of updates')

pyplot.tight_layout()
pyplot.show()