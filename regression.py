import pandas as pd
import numpy as np

#iris dataset is for classification, but let's try regression on it
#4 features: predict 4th feature from the first 3

INPUT_PATH = "iris.data"
#NUM_FEATURES = 3
NUM_CLASSES = 3
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
INITIAL_ETA = 0.03 #0.03 and lower works, but 0.1 too big -> weights go to large negative values, big error
ETA_DIVISION = 10
EPOCH_LENGTH = 1000
NUM_EPOCHS = 10
NUMERICAL_GRAD_DELTA = pow(10,-10)#pow(10,-15)

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
#dot product of point and weight generalizes to matrix multiplication of points vector with weights
#this also works when weights has multiple columns for multiple features
def predict_linear_all(points, weights):
	return np.matmul(points, weights)

#pick the best model weights for the data
def train_gradient_descent(features, labels, error_func, grad_func, initial_eta, eta_divide, epoch_length, num_epochs, grad_to_compare=None):
	#set initial weights- all 0, all 1 or something like that
	#loop over num_epochs:
		#loop over epoch length
			#compute gradient on weights, features, labels
			#move weights by gradient times learning rate
			#optionally log once per iteration: weights, error
		#or log once per epoch instead: weights, error
		#set learning rate = previous / eta_divide
	#return weights

	#weights = np.zeros((NUM_FEATURES+1, 1)) #initialize weights at 0 - does it matter where we start?

	#weights matri: map dimensions of one point(3 with augmented) to dimensions of label(2)
	label_dimension = labels.shape[1]
	feature_dimension = features.shape[1]
	weights = np.zeros((feature_dimension, label_dimension))

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

		if grad_to_compare:
			print("\nend of epoch gradient based on current gradient "+str(grad_func)+":\n"+str(grad_func(features, labels, weights)))
			print("\nend of epoch gradient based on other gradient "+str(grad_to_compare)+":\n"+str(grad_to_compare(features, labels, weights)))

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
	predictions = predict_linear_all(features, weights)
	errors = predictions - labels
	return np.matmul(features.transpose(), errors) / features.shape[0]

#if we don't know the gradient of some function, can directly measure
def numerical_gradient_of(lossfunc):
	return lambda features, labels, weights: numerical_gradient(lossfunc, features, labels, weights)

#gradient of some loss function with respect to the weights
#must be a lossfunc form with weight argument like mean_squared_loss_weights(features, labels, weights). not mean_squared_loss(predictions, labels)
def numerical_gradient(lossfunc, features, labels, weights):
	delta = NUMERICAL_GRAD_DELTA 
	grad = np.ones(weights.shape)
	loss_base = lossfunc(features, labels, weights)

	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			#take derivate with respect to weight[i]
			adjusted_weights = weights.copy()
			adjusted_weights[i][j] += delta
			value = (lossfunc(features, labels, adjusted_weights) - loss_base) / delta
			#print("value is "+str(value))
			#grad[i][j] = 1
			grad[i][j] = value
	return grad

#multiple targets: now weights has 2 dimensions
#can still use same prediction function, but loss is different
def multi_target_mean_squared_loss_weights(features, labels, weights):
	predictions = predict_linear_all(features, weights)
	return multi_target_mean_squared_loss(predictions, labels)

def multi_target_mean_squared_loss(predictions, labels):	
	if predictions.shape != labels.shape:
		raise ValueError("number or dimensionality of predictions does not match number of labels")
	n = len(predictions)
	errors = predictions - labels
	return sum([length_squared(predictions[i] - labels[i]) for i in range(n)]) / (2*n)

#gradient for multi target means squared loss with respect to weights.
#derivatives separate between the different target labels, so this is just a combination of the single target gradient for each target.
def multi_target_mean_squared_gradient(features, labels, weights):
	num_feature_vecs = features.shape[0]
	num_label_vecs = labels.shape[0]
	num_features = features.shape[1]
	num_labels = labels.shape[1]

	if num_feature_vecs	 != num_label_vecs	:
		raise ValueError("number of features and labels doesnt match")
	if weights.shape[0] != num_features:
		raise ValueError("weights dim 0 doesn't match num features")
	if weights.shape[1] != num_labels:
		raise ValueError("weights dim 1 doesn't match num labels")

	grad = np.zeros(weights.shape)
	for i in range(num_labels):
		#do part of gradient for ith label:
		#label vector is ith column of features
		#weights are ith column of weights only
		#features is entire feature matrix
		#gradient result is ith column of total gradient (matches shape of weights)
		grad[:,i] = mean_squared_gradient(features, labels[:,i], weights[:,i])
	return grad


#compute squared magnitude of a vector which is in 1d form
def length_squared(vec):
	return np.dot(vec, vec)

#using some fraction of data (randomly selected) as training set:
#train the model, test on remaining data, report statistics
def single_test_train(train_fraction, loss, grad, compare_grad=None):
	fulldata = pd.read_csv(INPUT_PATH, header=None)
	train_data = fulldata.sample(frac=train_fraction)
	test_data = fulldata.drop(train_data.index)

	train_X_aug, train_Y = divide_feature_label(train_data)
	test_X_aug, test_Y = divide_feature_label(test_data)

	#train, then test
	#should predicting and computing loss be combined? both involve predicting.

	weights = train_gradient_descent(train_X_aug, train_Y, loss, grad, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS, compare_grad)
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

#regression where target is two numbers - loss will be sum of squared vector differences
#use the first 2 features as features, second 2 as labels
#gradient and error functions must now handle Y of shape (N,2) instead of (N,1)
def two_numerical_targets_regression(train_fraction, loss, grad, compare_grad=None):
	fulldata = pd.read_csv(INPUT_PATH, header=None)
	train_data = fulldata.sample(frac=train_fraction)
	test_data = fulldata.drop(train_data.index)

	train_X_aug, train_Y = divide_2_feature_2_label(train_data)
	test_X_aug, test_Y = divide_2_feature_2_label(test_data)

	#train, then test
	#should predicting and computing loss be combined? both involve predicting.

	weights = train_gradient_descent(train_X_aug, train_Y, loss, grad, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS, compare_grad)
	predictions_train = predict_linear_all(train_X_aug, weights)

	train_loss = loss(train_X_aug, train_Y, weights)
	print("\nloss on training set: "+str(train_loss))
	print("truth - prediction")
	print(np.concatenate((train_Y, predictions_train), axis = 1))

	predictions_test = predict_linear_all(test_X_aug, weights)
	test_loss = loss(test_X_aug, test_Y, weights)
	print("\nloss on test set: "+str(test_loss))
	print("truth - prediction")
	print(np.concatenate((test_Y, predictions_test), axis = 1))

#classify with linear model
#classes go to "one hot" vectors as the features. to predict, take highest scoring class
#actually we need only n-1 vectors for n classes - is saving a variable worth more complicated equation?

#split into two classes: then can use single target variable with +/- 1.
def classify_binary(train_fraction, loss, grad, compare_grad=None):
	#data format: f1, f2, f3, f4, class
	#classs into 
	fulldata = pd.read_csv(INPUT_PATH, header=None)
	train_data = fulldata.sample(frac=train_fraction)
	test_data = fulldata.drop(train_data.index)

	train_X_aug, train_Y = divide_classification_binary(train_data)
	test_X_aug, test_Y = divide_classification_binary(test_data)

	weights = train_gradient_descent(train_X_aug, train_Y, loss, grad, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS, compare_grad)
	predictions_train = predict_linear_all(train_X_aug, weights)

	train_loss = loss(train_X_aug, train_Y, weights)
	print("\nloss on training set: "+str(train_loss))
	print("truth - prediction")
	print(np.concatenate((train_Y, predictions_train), axis = 1))

	predictions_test = predict_linear_all(test_X_aug, weights)
	test_loss = loss(test_X_aug, test_Y, weights)
	print("\nloss on test set: "+str(test_loss))
	print("truth - prediction")
	print(np.concatenate((test_Y, predictions_test), axis = 1))

#class is the label, so all 4 numbers are features
def divide_classification_binary(data):
	classes = data.pop(4)
	classes.reset_index(drop=True, inplace=True) #need this to get by index
	n = classes.size
	#0-1 vector for setosa or not.
	Y = np.zeros((n,1))
	for i in range(n):
		if classes[i] == "Iris-setosa":
			Y[i] = 1
	X = np.array(data)
	X_aug = augment(X)
	return (X_aug, Y)

#whatever statistics we want: accuracy etc 
def statistics():
	return

#one-hot classification into three classes
def classify_one_hot(train_fraction, loss, grad, compare_grad=None):
	fulldata = pd.read_csv(INPUT_PATH, header=None)
	train_data = fulldata.sample(frac=train_fraction)
	test_data = fulldata.drop(train_data.index)

	train_X_aug, train_Y = divide_classification_one_hot(train_data)
	test_X_aug, test_Y = divide_classification_one_hot(test_data)

	weights = train_gradient_descent(train_X_aug, train_Y, loss, grad, INITIAL_ETA, ETA_DIVISION, EPOCH_LENGTH, NUM_EPOCHS, compare_grad)
	predictions_train = predict_linear_all(train_X_aug, weights)

	train_loss = loss(train_X_aug, train_Y, weights)
	print("\nloss on training set: "+str(train_loss))
	print("truth - prediction")
	print(np.concatenate((train_Y, predictions_train), axis = 1))

	predictions_test = predict_linear_all(test_X_aug, weights)
	test_loss = loss(test_X_aug, test_Y, weights)
	print("\nloss on test set: "+str(test_loss))
	print("truth - prediction")
	print(np.concatenate((test_Y, predictions_test), axis = 1))

#class is the label, so all 4 numbers are features
def divide_classification_one_hot(data):
	classes = data.pop(4)
	classes.reset_index(drop=True, inplace=True) #need this to get by index
	n = classes.size
	#one-hot vector for the classes
	Y = np.zeros((n, NUM_CLASSES))
	for i in range(n):
		index = CLASS_NAMES.index(classes[i])
		Y[i][index] = 1
	X = np.array(data)
	X_aug = augment(X)
	return (X_aug, Y)


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

#feature is first 2 numbers, label is next 2. still drop the flower classes
#features mapping by indes: 0,1 -> 2,3	with 4 discarded
def divide_2_feature_2_label(data):
	classes = data.pop(4) #currently not using this
	Y_2 = to_2d_column_vec(np.array(data.pop(3)))
	Y_1 = to_2d_column_vec(np.array(data.pop(2)))
	Y = np.concatenate((Y_1,Y_2), axis = 1)
	X = np.array(data)
	X_aug = augment(X)
	return (X_aug, Y)

def main():
	#single_test_train(0.7, mean_squared_loss_weights, numerical_gradient_of(mean_squared_loss_weights), compare_grad = mean_squared_gradient)
	#single_test_train(0.7, mean_squared_loss_weights, mean_squared_gradient)
	#two_numerical_targets_regression(0.7, multi_target_mean_squared_loss_weights, numerical_gradient_of(multi_target_mean_squared_loss_weights))#, compare_grad = multi_target_mean_squared_gradient)
	#two_numerical_targets_regression(0.7, multi_target_mean_squared_loss_weights, multi_target_mean_squared_gradient, compare_grad = numerical_gradient_of(multi_target_mean_squared_loss_weights))
	#classify_binary(0.7, mean_squared_loss_weights, mean_squared_gradient)
	classify_one_hot(0.7, multi_target_mean_squared_loss_weights, multi_target_mean_squared_gradient)
