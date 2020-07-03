import pandas as pd
import numpy as np

#iris dataset is for classification, but let's try regression on it
#4 features: predict 4th feature from the first 3

INPUT_PATH = "iris.data"
NUM_FEATURES = 3
INITIAL_ETA = 1
ETA_DIVISION = 2
EPOCH_LENGTH = 100
NUM_EPOCHS = 10

def load()
	features = np.array(pd.read_csv(INPUT_PATH, usecols = [i for i in range(NUM_FEATURES)]))
	labels = np.array(pd.read_csv(INPUT_PATH, usecols = [NUM_FEATURES]))
	return (features, labels)

#predict point by linear model:
#y = w0 + w1 *x1 + w2 *x2 ... + wn *xn
#y = w dot x, where x is augmented: x = [1, x1, x2 ... xn] , w = [20 ... wn] 
def predict_linear(point, weights):
	return np.dot(point, weights)

#pick the best model weights for the data
def train_gradient_descent(features, labels, error_func, grad_func, initial_eta, eta_divide, epoch_length, num_epochs):
	return

#return loss for some model weights on the features and labels
def mean_squared_loss(features, labels, weights):
	return

#gradient for the mean squared error with respect to weights, at the current feature/label/weight position
def mean_squared_gradient(features, labels, weights):
	return

def main():
	features, labels = load()
	weights = train_gradient_descent(features, labels, mean_squared_loss, mean_squared_gradient, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS)
	