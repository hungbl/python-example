import pandas
from matplotlib import pyplot
import numpy
from perceptron import Perceptron
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(numpy.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = numpy.meshgrid(numpy.arange(x1_min, x1_max, resolution),
                           numpy.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(numpy.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    pyplot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(numpy.unique(y)):
        pyplot.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

df = pandas.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
tail = df.tail()
y = df.iloc[0:100, 4].values
y = numpy.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

#misclassification
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
pyplot.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
pyplot.xlabel('Epochs')
pyplot.ylabel('Number of updates')
pyplot.tight_layout()
pyplot.show()

#scatterplot
# pyplot.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
# pyplot.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plot_decision_regions(X, y, classifier=ppn)
pyplot.xlabel('sepal length [cm]')
pyplot.ylabel('petal length [cm]')
pyplot.legend(loc='upper left')
pyplot.tight_layout()
pyplot.show()