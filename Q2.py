from helper import *
import numpy as np 
import random

LOGISTIC_DATA_FILE = './HW#01_Datasets/Logistic/data_logistic.mat'
COLORS = {0: 'green', 1: 'red'}
ITERATIONS_COUNT = 800000
MINI_BATCH_SIZE = 10
LEARNING_RATE = 0.005
THRESHOLD = 0.5

def read_the_data():
	data = read_mat_file(LOGISTIC_DATA_FILE, 'logistic_data')
	return data

def calculate_sigmoid(x):
	return 1/(1+ np.exp(-x))

def calculate_gradient(X, Y, h):
	gradient = np.dot(X.T, (h-Y))/Y.size
	return gradient

def calculate_loss(h, Y):
	loss = (-Y * np.log(h) - (1 - Y) * np.log(1 - h))
	loss = sum(loss)/len(Y)
	return loss

def calculate_h(X, theta):
	z = np.dot(X, theta)
	h = calculate_sigmoid(z)
	return h

def logistic_regression(X, Y):
	theta = [0.1 , -0.1]
	cost_x, cost_y = [], []
	for iteration in range(0, ITERATIONS_COUNT):
		h = calculate_h(X, theta)
		gradient = calculate_gradient(X, Y, h)
		theta = theta-LEARNING_RATE*gradient
		
		if iteration%1000==0:
			h = calculate_h(X, theta)
			cost = calculate_loss(h, Y)
			cost_x.append(iteration)
			cost_y.append(cost)
			if iteration%100000==0:
				print('Loss at iteration {} is {}'.format(iteration, cost))
	draw_simple_connected_plot(cost_x, cost_y, 'number of iterations', 'cost', 'cost function')
	return theta

def logistic_regression_decreasing_learning_rate(X, Y):
	theta = [0, 0]
	cost_x, cost_y = [], []
	lr = 4
	for iteration in range(0, ITERATIONS_COUNT):
		h = calculate_h(X, theta)
		gradient = calculate_gradient(X, Y, h)
		theta = theta-lr*gradient
		
		lr *= 0.99
		if iteration%1000==0:
			h = calculate_h(X, theta)
			cost = calculate_loss(h, Y)
			cost_x.append(iteration)
			cost_y.append(cost)
			if iteration%100000==0:
				print('Loss at iteration {} is {}'.format(iteration, cost))
	draw_simple_connected_plot(cost_x, cost_y, 'number of iterations', 'cost', 'cost function')
	return theta

def pick_the_batch(X, Y):
	i = 0
	batch_X = []
	batch_Y = []
	picked_indexs = []
	while i<MINI_BATCH_SIZE:
		ind = random.randint(0, len(X)-1)
		if not ind in picked_indexs:
			i += 1
			batch_X.append(X[ind])
			batch_Y.append(Y[ind])
			picked_indexs.append(ind)
	batch_X = np.array(batch_X)
	batch_Y = np.array(batch_Y)
	return batch_X, batch_Y

def  logistic_regression_mini_batch(X, Y):
	theta = [0, 0]
	cost_x, cost_y = [], []
	for iteration in range(0, ITERATIONS_COUNT):
		batch_X, batch_Y = pick_the_batch(X, Y)

		h = calculate_h(batch_X, theta)
		gradient = calculate_gradient(batch_X, batch_Y, h)
		theta = theta-LEARNING_RATE*gradient

		if iteration%1000==0:
			h = calculate_h(X, theta)
			cost = calculate_loss(h, Y)
			cost_x.append(iteration)
			cost_y.append(cost)
			if iteration%100000==0:
				print('Loss at iteration {} is {}'.format(iteration, cost))
	draw_simple_connected_plot(cost_x, cost_y, 'number of iterations', 'cost', 'cost function')
	return theta

def divide_data(data):
	mean_of_first_att = 0
	mean_of_second_att = 0
	for inst in data:
		mean_of_first_att += inst[0]
		mean_of_second_att += inst[1]
	mean_of_first_att = mean_of_first_att/len(data)
	mean_of_second_att = mean_of_second_att/len(data)
	X  = []
	Y = []
	for inst in data:
		X.append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
		Y.append(inst[2])
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def calculate_accuracy(theta, X, Y):
	h = calculate_h(X, theta)
	prediction = h>=THRESHOLD
	result = prediction==Y
	correct = 0
	for r in result:
		if r==True:
			correct+=1
	print('The accuracy is: {}%'.format(100*correct/Y.size))

def batch_logistic_regression(data):
	X, Y = divide_data(data)
	theta = logistic_regression(X, Y)
	calculate_accuracy(theta, X, Y)

def batch_decreasing_learning_rate_logistic_regression(data):
	X, Y = divide_data(data)
	theta = logistic_regression_decreasing_learning_rate(X, Y)
	calculate_accuracy(theta, X, Y)

def mini_batch_logistic_regression(data):
	X, Y = divide_data(data)
	theta = logistic_regression_mini_batch(X, Y)
	calculate_accuracy(theta, X, Y)

def divide_data_5_fold(data):
	mean_of_first_att = 0
	mean_of_second_att = 0
	for inst in data:
		mean_of_first_att += inst[0]
		mean_of_second_att += inst[1]
	mean_of_first_att = mean_of_first_att/len(data)
	mean_of_second_att = mean_of_second_att/len(data)
	X  = {'0':[], '1':[], '2':[], '3':[], '4':[]}
	Y = {'0':[], '1':[], '2':[], '3':[], '4':[]}
	for ind, inst in enumerate(data):
		if ind<len(data)/5:
			X['0'].append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
			Y['0'].append(inst[2])
		elif ind<2*len(data)/5:
			X['1'].append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
			Y['1'].append(inst[2])
		elif ind<3*len(data)/5:
			X['2'].append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
			Y['2'].append(inst[2])
		elif ind<4*len(data)/5:
			X['3'].append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
			Y['3'].append(inst[2])
		else:
			X['4'].append([(inst[0]-mean_of_first_att)/100, (inst[1]-mean_of_second_att)/100])
			Y['4'].append(inst[2])
	for k in X.keys():
		X[k] = np.array(X[k])
		Y[k] = np.array(Y[k])
	return X, Y

def append(list, element):
	return np.concatenate((list, element), axis=0) if not list.size == 0 else element

def pick_the_correct_folds(X, Y, iteration):
	test_fold_number = str(iteration%5)
	test_X = X[test_fold_number]
	test_Y = Y[test_fold_number]
	train_X = np.array([])
	train_Y = np.array([])
	for k in X.keys():
		if not k==test_fold_number:
			train_X = append(train_X, X[k])
			train_Y = append(train_Y, Y[k])
	return train_X, train_Y, test_X, test_Y

def logistic_regression_5_fold(X, Y):
	theta = [0, 0]
	cost_x, cost_y = [], []
	for iteration in range(0, ITERATIONS_COUNT):
		train_X, train_Y, test_X, test_Y= pick_the_correct_folds(X, Y, iteration)

		h = calculate_h(train_X, theta)
		gradient = calculate_gradient(train_X, train_Y, h)
		theta = theta-LEARNING_RATE*gradient

		if iteration%10003==0:
			h = calculate_h(test_X, theta)
			cost = calculate_loss(h, test_Y)
			cost_x.append(iteration)
			cost_y.append(cost)
			print('Loss at iteration {} is {}'.format(iteration, cost))
	draw_simple_connected_plot(cost_x, cost_y, 'number of iterations', 'cost', 'cost function')
	return theta

def calculate_loss_regularized(h, Y, W, LAMBDA):
	sum_w_2 = sum([x**2 for x in W])
	loss = (-Y * np.log(h) - (1 - Y) * np.log(1 - h))
	loss = sum(loss)
	loss += (LAMBDA/2)*sum_w_2
	loss = loss/len(Y)
	return loss

def calculate_gradient_regularized(X, Y, h, theta, LAMBDA):
	gradient = np.dot(X.T, (h-Y))
	gradient += np.dot(LAMBDA,theta)
	gradient = gradient/Y.size
	return gradient

def logistic_regression_5_fold_regularized(X, Y, LAMBDA):
	theta = [0, 0]
	for iteration in range(0, ITERATIONS_COUNT):
		train_X, train_Y, test_X, test_Y= pick_the_correct_folds(X, Y, iteration)

		h = calculate_h(train_X, theta)
		gradient = calculate_gradient_regularized(train_X, train_Y, h, theta, LAMBDA)
		theta = theta-LEARNING_RATE*gradient

		if iteration%100003==0:
			h = calculate_h(test_X, theta)
			cost = calculate_loss_regularized(h, test_Y, theta, LAMBDA)
			print('Loss at iteration {} is {}'.format(iteration, cost))
	return theta


def five_fold_logistic_regression(data):
	X, Y = divide_data_5_fold(data)
	theta = logistic_regression_5_fold(X, Y)
	acc_X, acc_Y = divide_data(data)
	calculate_accuracy(theta, acc_X, acc_Y)

def answer_question_3(data):
	for power in range(-5,5):
		LAMBDA = 10**power
		print('LAMBDA values = ', LAMBDA)
		X, Y = divide_data_5_fold(data)
		theta = logistic_regression_5_fold_regularized(X, Y, LAMBDA)
		acc_X, acc_Y = divide_data(data)
		calculate_accuracy(theta, acc_X, acc_Y)

def answer_question_2(data):
	print('Batch Logistics Regression')
	batch_logistic_regression(data)
	print('Decreasing Learning Rate')
	batch_decreasing_learning_rate_logistic_regression(data)
	print('Mini-Batch Logistics Regression')
	mini_batch_logistic_regression(data)
	print('Using 5-folded cross validation')
	five_fold_logistic_regression(data)
	
def answer_question_1(data):
	x = []
	y = []
	color = []
	for row in data:
		x.append(float(row[0]))
		y.append(float(row[1]))
		color.append(COLORS[int(row[2])])

	x_label = 'first attribute'
	y_label = 'second attribute'
	title = 'sickness plot (green means healthy, red means sick)'
	draw_scatter_plot(x, y, color, x_label, y_label, title)

def see_the_sklearn_answer(data):
	from sklearn.linear_model import LogisticRegression
	X, Y = divide_data(data)
	logisticRegr = LogisticRegression()
	logisticRegr.fit(X, Y)
	score = logisticRegr.score(X, Y)
	calculate_accuracy(np.array(logisticRegr.coef_[0]), X, Y)

def main():
	data = read_the_data()
	answer_question_1(data)
	answer_question_2(data)
	answer_question_3(data)
	see_the_sklearn_answer(data)

if __name__ == '__main__':
	main()