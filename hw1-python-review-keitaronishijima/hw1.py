# Python 3.7
import numpy as np  # Used to perform efficient (and convenient) array and matrix operations.
from sklearn import datasets  # Used to load standard datasets.
from matplotlib import pyplot as plt  # Used to create plots.
import math  # Used for trigonometric functions, log, pow, etc.


def graph_iris_data():
	''' 
	function for graphing iris data. Fill this out to get a scatter plot of iris datapoints!
	'''

	iris = datasets.load_iris()
	data = iris.data
	xs = data[:, 0]
	ys = data[:, 1]

	# [TODO] plot iris data here
	plt.scatter(xs, ys, c=iris.target)
	plt.title("Made by: B01644365")
	plt.xlabel(iris.feature_names[0])
	plt.ylabel(iris.feature_names[1])
	plt.show()


def graph_series_data():
	'''
	function for graphing series data
	'''

	xs = np.arange(100) * .5 - 10  # Creates a list of 100 values in intervals of .5, starting at -10.
	# Numpy makes it very easy (and efficient) to do elementwise operations on large datasets.
	y1s = [x*x for x in xs]  # If you are unfamiliar, this is called Python's "List Comprehension Syntax"
	y2s = [math.sin(x)*x*x for x in xs]
	# https://matplotlib.org/users/pyplot_tutorial.html is a good starting point for more info.
	
	# [TODO] plot series data here
	plt.plot(xs, y1s, '.r', label = "x^x")
	plt.plot(xs, y2s, '-b', label = "sin(x)*x*x")
	plt.legend()
	plt.title("Made by: B01644365")



	plt.show()


if __name__ == '__main__':
	graph_iris_data()
	graph_series_data()
