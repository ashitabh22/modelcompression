from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import np_utils

# Import other necessary packages
import numpy as np



def get_data_cifar(num_classes=10):
	"""
	Get the CIFAR dataset.

	Parameters:
		None
	Returns:
		train_data - training data split
		train_labels - training labels
		test_data - test data split
		test_labels - test labels
	"""
	print('[INFO] Loading the CIFAR10 dataset...')
	(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

	# Transform labels to one hot labels
	# Example: '0' will become [1, 0, 0, 0, 0, 0, 0, 0, 0]
	#          '1' will become [0, 1, 0, 0, 0, 0, 0, 0, 0]
	#          and so on...
	train_labels = np_utils.to_categorical(train_labels, num_classes)
	test_labels = np_utils.to_categorical(test_labels, num_classes)

	# Change type and normalize data
	train_data = train_data.astype('float32')
	test_data = test_data.astype('float32')
	train_data /= 255
	test_data /= 255

	return train_data, train_labels, test_data, test_labels


def get_data_mnist(num_classes=10):
	"""
	Get the MNIST dataset.

	Will download dataset if first time and will be downloaded
	to ~/.keras/datasets/mnist.npz
	Parameters:
		None
	Returns:
		train_data - training data split
		train_labels - training labels
		test_data - test data split
		test_labels - test labels
	"""
	print('[INFO] Loading the MNIST dataset...')
	(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

	# Reshape the data from (samples, height, width) to
	# (samples, height, width, depth) where depth is 1 channel (grayscale)
	train_data = train_data[:, :, :, np.newaxis]
	test_data = test_data[:, :, :, np.newaxis]

	# Normalize the data
	train_data = train_data / 255.0
	test_data = test_data / 255.0

	# Transform labels to one hot labels
	# Example: '0' will become [1, 0, 0, 0, 0, 0, 0, 0, 0]
	#          '1' will become [0, 1, 0, 0, 0, 0, 0, 0, 0]
	#          and so on...
	train_labels = np_utils.to_categorical(train_labels, num_classes)
	test_labels = np_utils.to_categorical(test_labels, num_classes)

	return train_data, train_labels, test_data, test_labels

# Import other necessary packages
