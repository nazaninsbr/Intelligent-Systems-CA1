from helper import *
import numpy as np 
import random 
import copy 
import sys
import math
import matplotlib.pyplot as plt
import os


ITERATION = 200
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

def findMeanOfEverything(cluster):
	if len(cluster)==0:
		return [0, 0, 0, 0]
	s = []
	for fieldId in range(len(cluster[0][1])):
		s.append(0)
		for ins in cluster:
			s[-1] += ins[1][fieldId]
		s[-1] /= len(cluster)
	return s

def clusterBasedOnEveryThingWithEuclideanDistance(server_data, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		server_data_copy = copy.deepcopy(server_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(server_data_copy)-1)
			centers[i] = server_data_copy[ind]
			clusters[i] = []
			clusters[i].append([ind, server_data_copy[ind]])
			del server_data_copy[ind]

		for ind, ins in enumerate(server_data_copy):
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateDist(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append([ind, ins])

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateDist(clusters[clusterNum][insId][1], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateDist(clusters[clusterNum][insId][1], centers[clusterNum]))**2

			cost_func = cost_func / len(server_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateDist(val[1], centers[centerId])

		inner_dist = inner_dist/len(server_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateDist(val[1], centers[centerId2])

		outer_dist = outer_dist/len(server_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		for clusterNum in clusters.keys():
			fileName = str(k_num)+'_'+str(clusterNum)+'_Kcluster.txt'
			try:
				os.remove(fileName)
			except OSError:
				pass
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCenters, resultingClusters

def clusterBasedOnEveryThingWithManhattanDistance(server_data, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		server_data_copy = copy.deepcopy(server_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(server_data_copy)-1)
			centers[i] = server_data_copy[ind]
			clusters[i] = []
			clusters[i].append([ind, server_data_copy[ind]])
			del server_data_copy[ind]

		for ind, ins in enumerate(server_data_copy):
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateManhattanDist(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append([ind, ins])

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateManhattanDist(clusters[clusterNum][insId][1], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateManhattanDist(clusters[clusterNum][insId][1], centers[clusterNum]))**2

			cost_func = cost_func / len(server_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateManhattanDist(val[1], centers[centerId])

		inner_dist = inner_dist/len(server_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateManhattanDist(val[1], centers[centerId2])

		outer_dist = outer_dist/len(server_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		for clusterNum in clusters.keys():
			fileName = str(k_num)+'_'+str(clusterNum)+'_Kcluster_Manhattan_dist.txt'
			try:
				os.remove(fileName)
			except OSError:
				pass
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCenters, resultingClusters

def clusterBasedOnEveryThingWithCosineSimilarity(server_data, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		server_data_copy = copy.deepcopy(server_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(server_data_copy)-1)
			centers[i] = server_data_copy[ind]
			clusters[i] = []
			clusters[i].append([ind, server_data_copy[ind]])
			del server_data_copy[ind]

		for ind, ins in enumerate(server_data_copy):
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateCosineSimilarity(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append([ind, ins])

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) <= insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateCosineSimilarity(clusters[clusterNum][insId][1], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateCosineSimilarity(clusters[clusterNum][insId][1], centers[clusterNum]))**2

			cost_func = cost_func / len(server_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateCosineSimilarity(val[1], centers[centerId])

		inner_dist = inner_dist/len(server_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateCosineSimilarity(val[1], centers[centerId2])

		outer_dist = outer_dist/len(server_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		for clusterNum in clusters.keys():
			fileName = str(k_num)+'_'+str(clusterNum)+'_Kcluster_cosine_sim.txt'
			try:
				os.remove(fileName)
			except OSError:
				pass
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCenters, resultingClusters


def answer_to_question_123_no_cross_validation(data, labels, k):
	print('Algo with distance #1')
	myCenters1, myClusters1 = clusterBasedOnEveryThingWithEuclideanDistance(data, k)
	majorities1 = calculate_cluster_majority(myClusters1, labels)
	stats1 = calculate_how_many_wrongly_classified(myClusters1, labels, majorities1)
	for kVal in stats1.keys():
		print('with Euclidean distance for k = {}, {} of {} instances wrongly clustered'.format(kVal, stats1[kVal][1], stats1[kVal][0]))
	print('Algo with distance #2')
	myCenters2, myClusters2 = clusterBasedOnEveryThingWithManhattanDistance(data, k)
	majorities2 = calculate_cluster_majority(myClusters2, labels)
	stats2 = calculate_how_many_wrongly_classified(myClusters2, labels, majorities2)
	for kVal in stats1.keys():
		print('with manhattan distance for k = {}, {} of {} instances wrongly clustered'.format(kVal, stats2[kVal][1], stats2[kVal][0]))
	print('Algo with distance #3')
	myCenters3, myClusters3 = clusterBasedOnEveryThingWithCosineSimilarity(data, k)
	majorities3 = calculate_cluster_majority(myClusters3, labels)
	stats3 = calculate_how_many_wrongly_classified(myClusters3, labels, majorities3)
	for kVal in stats1.keys():
		print('with cosine similarity for k = {}, {} of {} instances wrongly clustered'.format(kVal, stats3[kVal][1], stats3[kVal][0]))

def find_cluster_centers(data, labels, k):
	seen_classes, centers = [], []
	for ind, ins in enumerate(data):
		if len(centers)==k:
			break
		if not labels[ind] in seen_classes:
			seen_classes.append(labels[ind])
			centers.append(ind)
	return centers

def clusterGivenInitialCenters(initial_centers, server_data, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		server_data_copy = copy.deepcopy(server_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = initial_centers[i]
			centers[i] = server_data_copy[ind]
			clusters[i] = []
			clusters[i].append([ind, server_data_copy[ind]])
			del server_data_copy[ind]

		for ind, ins in enumerate(server_data_copy):
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateDist(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append([ind, ins])

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateDist(clusters[clusterNum][insId][1], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateDist(clusters[clusterNum][insId][1], centers[clusterNum]))**2

			cost_func = cost_func / len(server_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateDist(val[1], centers[centerId])

		inner_dist = inner_dist/len(server_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateDist(val[1], centers[centerId2])

		outer_dist = outer_dist/len(server_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		for clusterNum in clusters.keys():
			fileName = str(k_num)+'_'+str(clusterNum)+'_Kcluster_initial_centers.txt'
			try:
				os.remove(fileName)
			except OSError:
				pass
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCenters, resultingClusters


def find_most_seen(count_each_class):
	if count_each_class[0]>=count_each_class[1] and count_each_class[0]>=count_each_class[2]:
		return 1
	if count_each_class[1]>=count_each_class[0] and count_each_class[1]>=count_each_class[2]:
		return 2
	if count_each_class[2]>=count_each_class[1] and count_each_class[2]>=count_each_class[0]:
		return 3

def calculate_cluster_majority(myClusters, labels):
	majority = {}
	for kVal in myClusters.keys():
		majority[kVal] = {}
		for classNumber in myClusters[kVal].keys():
			count_each_class = [0, 0, 0]
			for ins in myClusters[kVal][classNumber]:
				count_each_class[labels[ins[0]][0]-1] +=1
			majority[kVal][classNumber] = find_most_seen(count_each_class)
	return majority

def calculate_how_many_wrongly_classified(myClusters, labels, majority):
	stats = {}
	for kVal in myClusters.keys():
		all_instances, wrongly_clustered = 0, 0
		for classNumber in myClusters[kVal].keys(): 
			for ins in myClusters[kVal][classNumber]:
				all_instances += 1
				if not labels[ins[0]][0]==majority[kVal][classNumber]:
					# print("ins: {}, label: {}, majority:{}".format(ins, labels[ins[0]][0], majority[kVal][classNumber]))
					wrongly_clustered += 1
		stats[kVal] = [all_instances, wrongly_clustered]
	return stats


def extra_answer_to_question_3(data, labels, k):
	initial_centers = find_cluster_centers(data, labels, k)
	myCenters, myClusters = clusterGivenInitialCenters(initial_centers, data, [3])
	majorities = calculate_cluster_majority(myClusters, labels)
	stats = calculate_how_many_wrongly_classified(myClusters, labels, majorities)
	print('{} of {} instances wrongly clustered'.format(stats[3][1], stats[3][0]))

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

def clusterBasedOnEveryThingWithEuclideanDistanceWithKFold(folds, labels, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		centers = {}
		training_set, test_set = create_train_test_folds(folds, 0)
		for i in range(k_num):
			ind = random.randint(0, len(training_set)-1)
			centers[i] = training_set[ind][1]

		for fold_num in range(6):
			# create the initial clusters
			clusters = {}
			training_set, test_set = create_train_test_folds(folds, fold_num)

			for ins in training_set:
				minDist = sys.maxsize
				clusterNum = -1
				for center in centers.keys():
					dist = calculateDist(ins[1], centers[center])
					if dist < minDist:
						minDist = dist
						clusterNum = center
				if not clusterNum in clusters.keys():
					clusters[clusterNum] = []
				clusters[clusterNum].append(ins)

			# run the clustering algo.
			for inter_num in range(ITERATION):
				# calculate new centers 
				for clusterNum in clusters.keys():
					mean = findMeanOfEverything(clusters[clusterNum])
					centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

				# reassign elements
				for clusterNum in clusters.keys():
					for insId in range(len(clusters[clusterNum])):
						if len(clusters[clusterNum]) == insId:
							break
						minDist = sys.maxsize
						newClusterNum = -1
						for center in centers.keys():
							dist = calculateDist(clusters[clusterNum][insId][1], centers[center])
							if dist < minDist:
								minDist = dist
								newClusterNum = center

						clusters[newClusterNum].append(clusters[clusterNum][insId])
						del clusters[clusterNum][insId]

				# cost function 
			cost_func = 0
			majorities = calculate_cluster_majority({k_num:clusters}, labels)
			print(majorities)
			testclusters = {k_num: {}}
			for ins in test_set:
				minDist = sys.maxsize
				clusterNum = -1
				for center in centers.keys():
					dist = calculateDist(ins[1], centers[center])
					if dist < minDist:
						minDist = dist
						clusterNum = center
				if not clusterNum in testclusters[k_num].keys():
					testclusters[k_num][clusterNum] = []
				testclusters[k_num][clusterNum].append(ins)
			print(testclusters)
			print(testclusters[k_num].keys())
			exit()
			stats = calculate_how_many_wrongly_classified(testclusters, labels, majorities)
			plt.scatter(fold_num, stats[k_num][0], color='black')

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateDist(val[1], centers[centerId])

		inner_dist = inner_dist/160

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateDist(val[1], centers[centerId2])

		outer_dist = outer_dist/160

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		plt.show()

	return resultingCenters, resultingClusters

def answer_to_question_123_with_cross_validation(data, labels, k):
	folds = create_6_folds(data)
	print('Algo with distance #1 using K-fold')
	myCenters1, myClusters1 = clusterBasedOnEveryThingWithEuclideanDistanceWithKFold(folds, labels, k)

def main():
	data, labels = get_the_data()
	answer_to_question_123_no_cross_validation(data.tolist(), labels.tolist(), [3, 5, 7, 9])
	extra_answer_to_question_3(data.tolist(), labels.tolist(), 3)
	answer_to_question_123_with_cross_validation(data.tolist(), labels.tolist(), [3, 5, 7, 9])


if __name__ == '__main__':
	main()