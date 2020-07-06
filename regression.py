import pandas as pd
import numpy as np

#iris dataset is for classification, but let's try regression on it
#4 features: predict 4th feature from the first 3

INPUT_PATH = "iris.data"
NUM_FEATURES = 3
INITIAL_ETA = 0.03 #0.03 and lower works, but 0.1 too big -> weights go to large negative values, big error
ETA_DIVISION = 10
EPOCH_LENGTH = 1000
NUM_EPOCHS = 5
NUMERICAL_GRAD_DELTA = pow(10,-15)

# def load():
# 	features = np.array(pd.read_csv(INPUT_PATH, usecols = [i for i in range(NUM_FEATURES)]))
# 	labels = np.array(pd.read_csv(INPUT_PATH, usecols = [NUM_FEATURES]))
# 	return (features, labels)

#numpy 1d array to numpy 2d column vector shape
def to_2d_column_vec(vec):
	return vec.reshape((vec.size,1))

#2d vec back to 1d vec
def to_1d_vec(vec):
	return vec.reshape(vec.size)

#predict point by linear model:
#y = w0 + w1 *x1 + w2 *x2 ... + wn *xn
#y = w dot x, where x is augmented: x = [1, x1, x2 ... xn] , w = [20 ... wn] 
def predict_linear_single(point, weights):
	return np.dot(point, weights)

#predict all the points. 
def predict_linear_all(points, weights):
	return np.matmul(points, weights)

#pick the best model weights for the data
def train_gradient_descent(features, labels, error_func, grad_func, initial_eta, eta_divide, epoch_length, num_epochs):
	#set initial weights- all 0, all 1 or something like that
	#loop over num_epochs:
		#loop over epoch length
			#compute gradient on weights, features, labels
			#move weights by gradient times learning rate
			#optionally log once per iteration: weights, error
		#or log once per epoch instead: weights, error
		#set learning rate = previous / eta_divide
	#return weights

	weights = np.zeros((NUM_FEATURES+1, 1)) #initialize weights at 0 - does it matter where we start?
	eta = initial_eta
	for epoch in range(num_epochs):
		
		print("\nepoch "+str(epoch)+":")
		print("\teta: "+str(eta))
		#print("\tloss before updates = "+str(loss))

		for i in range(epoch_length):
			grad = grad_func(features, labels, weights)
			weights -= eta*grad
			#print("\titeration "+str(i))
			#print("\t\tweights = "+str(to_1d_vec(weights)))
			#print("\t\tgrad = "+str(to_1d_vec(grad)))
		eta /= eta_divide
		loss = error_func(features, labels, weights)
		print("\tweights after update:\n"+str(weights))
		print("\tloss after update: "+str(loss))

	return weights

#mean squared loss: use the same coefficients as for gradient and MLE calculations
def mean_squared_loss(predictions, labels):
	if predictions.shape != labels.shape:
		raise ValueError("number of predictions does not match number of labels")
	n = len(predictions)
	#use numpy array operations instead of list comprehension?
	return sum([pow(predictions[i] - labels[i], 2) for i in range(n)]) / (2*n) 

#return loss for some model weights on the features and labels
def mean_squared_loss_weights(features, labels, weights):
	#TODO: can take an argument for the model, but for now just force it to be linear.
	predictions = predict_linear_all(features, weights)
	return mean_squared_loss(predictions, labels) 

#gradient for the mean squared error with respect to weights, at the current feature/label/weight position
def mean_squared_gradient(features, labels, weights):
	return weights #DUMMY

#if we don't know the gradient of some function, can directly measure
def empirical_gradient_of(lossfunc):
	return lambda features, labels, weights: empirical_gradient(lossfunc, features, labels, weights)

def empirical_gradient(lossfunc, features, labels, weights):
	delta = NUMERICAL_GRAD_DELTA
	grad = np.zeros(weights.shape)
	loss_base = lossfunc(features, labels, weights)
	for i in range(weights.size):
		#take derivate with respect to weight[i]
		adjusted_weights = weights.copy()
		adjusted_weights[i] += delta
		grad[i] = (lossfunc(features, labels, adjusted_weights) - loss_base) / delta
	return grad

#using some fraction of data (randomly selected) as training set:
#train the model, test on remaining data, report statistics
def single_test_train(train_fraction, loss, grad):
	fulldata = pd.read_csv(INPUT_PATH, header=None)
	train_data = fulldata.sample(frac=train_fraction)
	test_data = fulldata.drop(train_data.index)

	train_X_aug, train_Y = divide_feature_label(train_data)
	test_X_aug, test_Y = divide_feature_label(test_data)

	#train, then test
	#should predicting and computing loss be combined? both involve predicting.

	weights = train_gradient_descent(train_X_aug, train_Y, loss, grad, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS)
	predictions_train = predict_linear_all(train_X_aug, weights)

	train_loss = loss(train_X_aug, train_Y, weights)
	print("loss on training set: "+str(train_loss))
	print("truth - prediction")
	print(np.concatenate((train_Y, predictions_train), axis = 1))

	predictions_test = predict_linear_all(test_X_aug, weights)
	test_loss = loss(test_X_aug, test_Y, weights)
	print("loss on test set: "+str(test_loss))
	print("truth - prediction")
	print(np.concatenate((test_Y, predictions_test), axis = 1))

	return #should it return anything? maybe return the final weights and statistics

#whatever statistics we want: accuracy etc 
def statistics():
	return


#augmented form for calculations with bias: add a column of 1 at the beginning
#original is a numpy 2d array
def augment(original):
	extra = np.ones((original.shape[0],1))
	return np.concatenate((extra, original), axis = 1)

#leave-one-out test: predict one point point with model trained on all other points
#do this on all points, report average statistics
def leave_one_out(fulldata):
	fulldata = pd.read_csv("iris.data", header=None)
	for i in fulldata.index:
		row = fulldata.loc[[i]] #ith row. loc[i] makes it a column, loc[[i]] makes row
		rest = fulldata.drop(fulldata.index[i]) #returns a copy with row removed, does not affect original

		train_X_aug, train_Y = divide_feature_label(rest)
		test_X_aug, test_Y = divide_feature_label(row)

#split dataframe into x and y parts, augment x
def divide_feature_label(data):
	classes = data.pop(4) #currently not using this
	Y = to_2d_column_vec(np.array(data.pop(3)))
	X = np.array(data)
	X_aug = augment(X)
	return (X_aug, Y)

def main():
	single_test_train(0.7,mean_squared_loss_weights,empirical_gradient_of(mean_squared_loss_weights))