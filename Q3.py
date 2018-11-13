from helper import *
import numpy as np 
import random 
import copy 
import sys
import math
import matplotlib.pyplot as plt
import os


ITERATION = 200
P = 3
KNN_DATA_FILE = './HW#01_Datasets/KNN/data.mat'
KNN_LABELS_FILE = './HW#01_Datasets/KNN/labels.mat'

def get_the_data():
	data = read_mat_file(KNN_DATA_FILE, 'data2')
	labels = read_mat_file(KNN_LABELS_FILE, 'labels')
	return data, labels


def calculateDist(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += (ins[i] - center[i]) ** 2
	dist = math.sqrt(dist)
	return dist

def calculateMinkowskiDistance(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += abs(ins[i] - center[i]) ** P
	dist = dist**(1/float(P))
	return dist

def calculateChebyshevDistance(ins, center):
	max_dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist = abs(ins[i] - center[i])
		if dist>max_dist:
			max_dist = dist
	return max_dist

def calculateManhattanDist(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += abs(ins[i] - center[i])
	return dist

def calculateCosineSimilarity(ins, center):
	dot = 0
	norm_1 = 0
	norm_2 = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dot += ins[i]*center[i]
		norm_1 += (ins[i])**2
		norm_2 += (center[i])**2
	if norm_1==0 or norm_2==0:
		return 1
	return dot/(math.sqrt(norm_1)*math.sqrt(norm_2))

def create_6_folds(data):
	folds = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
	for ind, ins in enumerate(data):
		if ind<len(data)/6:
			folds[0].append([ind, ins])
		elif ind<2*len(data)/6:
			folds[1].append([ind, ins])
		elif ind<3*len(data)/6:
			folds[2].append([ind, ins])
		elif ind<4*len(data)/6:
			folds[3].append([ind, ins])
		elif ind<5*len(data)/6:
			folds[4].append([ind, ins])
		else:
			folds[5].append([ind, ins])
	return folds

def create_train_test_folds(folds, fold_num):
	test_set = folds[fold_num]
	training_set = []
	for x in folds.keys():
		if not x==fold_num:
			training_set.extend(folds[x])
	return training_set, test_set

def sort_based_on_distance(distances, indexes):  
	sorted_distances, sorted_indexes = copy.deepcopy(distances), copy.deepcopy(indexes)
	for i in range(1, len(distances)): 
  
		key = sorted_distances[i] 
		index_val = sorted_indexes[i]
		j = i-1
		while j >=0 and key < sorted_distances[j] : 
				sorted_distances[j+1] = sorted_distances[j]
				sorted_indexes[j+1] = sorted_indexes[j] 
				j -= 1
		sorted_distances[j+1] = key 
		sorted_indexes[j+1] = index_val
	return sorted_distances, sorted_indexes

def find_k_closest(x, training_set, k, distance_type):
	distances = []
	indexes = []
	for ind, ins in enumerate(training_set):
		if distance_type=='E':
			dist = calculateDist(ins[1], x[1])
		elif distance_type=='M':
			dist = calculateManhattanDist(ins[1], x[1])
		elif distance_type=='C':
			dist = calculateCosineSimilarity(ins[1], x[1])
		elif distance_type=='Minkowski':
			dist = calculateMinkowskiDistance(ins[1], x[1])
		elif distance_type=='Chebyshev':
			dist = calculateChebyshevDistance(ins[1], x[1])
		distances.append(dist)
		indexes.append(ind)

	sorted_distances, sorted_indexes = sort_based_on_distance(distances, indexes)

	nearerst = []
	for i in range(k):
		nearerst.append(training_set[sorted_indexes[i]])

	return nearerst

def find_most_seen(count_each_class):
	if count_each_class[1]>=count_each_class[2] and count_each_class[1]>=count_each_class[3]:
		return 1
	if count_each_class[2]>=count_each_class[1] and count_each_class[2]>=count_each_class[3]:
		return 2
	if count_each_class[3]>=count_each_class[1] and count_each_class[3]>=count_each_class[2]:
		return 3

def find_predicted_class(A, labels):
	count = {1:0, 2:0, 3:0}
	for ins in A:
		l = labels[ins[0]][0]
		count[l] += 1
	return find_most_seen(count)

def knn(folds, labels, k):
	print('distance type 1')
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for fold_num in range(6):
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for x in test_set:
				all_instances += 1
				A = find_k_closest(x, training_set, k_num, 'E')
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[x[0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified'.format(k_num, wrongly_classified, all_instances))

def knn_with_manhattan(folds, labels, k):
	print('distance type 2')
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for fold_num in range(6):
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for x in test_set:
				all_instances += 1
				A = find_k_closest(x, training_set, k_num, 'M')
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[x[0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified'.format(k_num, wrongly_classified, all_instances))

def knn_with_cosine_similarity(folds, labels, k):
	print('distance type 3')
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for fold_num in range(6):
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for x in test_set:
				all_instances += 1
				A = find_k_closest(x, training_set, k_num, 'C')
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[x[0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified'.format(k_num, wrongly_classified, all_instances))

def knn_with_Minkowski_distance(folds, labels, k):
	print('distance type 4')
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for fold_num in range(6):
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for x in test_set:
				all_instances += 1
				A = find_k_closest(x, training_set, k_num, 'Minkowski')
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[x[0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified'.format(k_num, wrongly_classified, all_instances))

def knn_with_Chebyshev_distance(folds, labels, k):
	print('distance type 5')
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for fold_num in range(6):
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for x in test_set:
				all_instances += 1
				A = find_k_closest(x, training_set, k_num, 'Chebyshev')
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[x[0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified'.format(k_num, wrongly_classified, all_instances))


def answer_to_question_12(data, labels, k):
	folds = create_6_folds(data)
	knn(folds, labels, k)
	knn_with_manhattan(folds, labels, k)
	knn_with_cosine_similarity(folds, labels, k)

def normalize_data(data):
	d = copy.deepcopy(data)
	m1, m2, m3, m4 = 0, 0, 0, 0
	for ins in d:
		m1 += ins[0]
		m2 += ins[1]
		m3 += ins[2]
		m4 += ins[3]

	m1, m2, m3, m4 = m1/len(d), m2/len(d), m3/len(d), m4/len(d)

	for ind, ins in enumerate(d):
		d[ind] = [(ins[0]-m1)/10, (ins[1]-m2)/10, (ins[2]-m3)/10, (ins[3]-m4)/10]
	return d


def method_1_for_q3(data, labels, k):
	print('after normalization')
	normalized_data = normalize_data(data)
	folds = create_6_folds(normalized_data)
	knn(folds, labels, k)
	knn_with_manhattan(folds, labels, k)
	knn_with_cosine_similarity(folds, labels, k)

def method_2_for_q3(data, labels, k):
	print('back to not normalized data')
	folds = create_6_folds(data)
	knn_with_Minkowski_distance(folds, labels, k)
	knn_with_Chebyshev_distance(folds, labels, k)

def main():
	data, labels = get_the_data()
	answer_to_question_12(data.tolist(), labels.tolist(), [3, 5, 7, 9])
	method_1_for_q3(data.tolist(), labels.tolist(), [3, 5, 7, 9])
	method_2_for_q3(data.tolist(), labels.tolist(), [3, 5, 7, 9])


if __name__ == '__main__':
	main()