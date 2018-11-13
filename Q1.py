from helper import *
import math 
from scipy.stats import norm
import pylab
import numpy as np

BAYESIAN_TRAIN_FILE = './HW#01_Datasets/Bayesian/data_train.mat'
BAYESIAN_TEST_FILE = './HW#01_Datasets/Bayesian/data_test.mat'

def get_the_data():
	train_data = read_mat_file(BAYESIAN_TRAIN_FILE, 'data_train')
	test_data = read_mat_file(BAYESIAN_TEST_FILE, 'data_test')
	return train_data, test_data

def separate_data_by_class(train_data):
	separated = {'0': [], '1': []}
	for x in train_data:
		separated[str(int(x[-1]))].append(x)
	return separated

def answer_to_question_1(separated_train_data):
	total_num = len(separated_train_data['0']) + len(separated_train_data['1'])
	num_of_not_healthy = len(separated_train_data['1'])
	sick_probability = num_of_not_healthy/total_num*100
	print("Probability of being sick = {}%".format(sick_probability))
	draw_bar_plot("Probability of being sick = {}%".format(sick_probability), "number of people in each class" ,  ['0', '1'], [total_num-num_of_not_healthy, num_of_not_healthy])
	return sick_probability/100

def calculate_mean(attribute_values):
	return sum(attribute_values)/len(attribute_values)

def calculate_standard_deviation(attribute_values):
	mean = calculate_mean(attribute_values)
	return math.sqrt(sum([(x-mean)**2 for x in attribute_values])/(len(attribute_values)-1))


def calculate_naive_bayes(all_data_from_one_class):
	return [(calculate_mean(attribute), calculate_standard_deviation(attribute)) for attribute in zip(*all_data_from_one_class)]

def plot_the_normal_distributions(training_result):
	i = 0
	for classNumber in training_result.keys():
		counter = 0 
		for dist in training_result[classNumber]:
			counter += 1
			i += 1
			plt.subplot(5, 2, i)
			x = np.linspace(-5, 5, 5000)
			mu = dist[0]
			sigma = dist[1]
			y = (1 / (np.sqrt(2 * np.pi * np.power(sigma, 2)))) * (np.power(np.e, -(np.power((x - mu), 2) / (2 * np.power(sigma, 2)))))
			if classNumber=='1':
				plt.plot(x,y, color='red')
			else:
				plt.plot(x,y, color='green')
			plt.title("distribution for feature {} of class {}".format(counter, classNumber))
	plt.show()

def train_the_model(separated_train_data):
	training_result = {}
	for class_number in separated_train_data.keys():
		training_result[class_number] = calculate_naive_bayes(separated_train_data[class_number])
	for classNumber in training_result.keys():
			training_result[classNumber].pop()
	plot_the_normal_distributions(training_result)
	return training_result

def calculate_normal_distribution_probability_function(inp, mean, standard_dev):
	power = -1*(((inp-mean)**2)/(2*(standard_dev)**2))
	factor = 1/(math.sqrt(2*math.pi)*standard_dev)
	return factor*math.exp(power)

def predict(x, training_result, sickness_probability):
	probabilities = {}
	for class_number in training_result.keys():
		att_count = -1
		probabilities[class_number] = 1
		for attribute_res in training_result[class_number]:
			att_count += 1
			mean = attribute_res[0]
			standard_dev = attribute_res[1]
			inp = x[att_count]
			probabilities[class_number] *= calculate_normal_distribution_probability_function(inp, mean, standard_dev)
		if class_number=='1':
			probabilities[class_number] *= sickness_probability
		else:
			probabilities[class_number] *= (1-sickness_probability)

	result = ''
	max_probability = 0
	for x in probabilities.keys():
		if result=='':
			max_probability = probabilities[x]
			result = x
		elif probabilities[x]>max_probability:
			max_probability = probabilities[x]
			result = x
	return int(result)


def predict_for_test_set(test_data, training_result, sickness_probability):
	correct = 0
	for x in test_data:
		correct_class = int(x[-1])
		x = x[:len(x)-1]
		prediction = predict(x, training_result, sickness_probability)
		if prediction==correct_class:
			correct+=1
	return correct



def answer_to_question_2(separated_train_data, test_data, sickness_probability):
	training_result = train_the_model(separated_train_data)
	number_of_correct_predictions = predict_for_test_set(test_data, training_result, sickness_probability) 
	print('Accuracy of the model is: {}%'.format(100*number_of_correct_predictions/len(test_data)))

def main():
	train_data, test_data = get_the_data()
	separated_train_data = separate_data_by_class(train_data)
	sickness_probability = answer_to_question_1(separated_train_data)
	answer_to_question_2(separated_train_data, test_data, sickness_probability)

if __name__ == '__main__':
	main()