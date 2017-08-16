import numpy as np
import random
import copy
import json

def getBest(pool):
	best_fitness = 0
	best_ind = None

	for individual in pool:
		fitness = individual.fitness
		if fitness >= best_fitness:
			best_fitness = fitness
			best_ind = individual

	if not best_ind:
		return random.choice(pool)
	return best_ind

def tournament(pool, k, tourn_size):
	chosen = []
	for i in range(k):
		pool = random.sample(pool, tourn_size)
		chosen.append(getBest(pool))
	return chosen

#Single Point Crossover
def crossover(weights1, weights2):
	weights1_new = copy.deepcopy(weights1)
	weights2_new = copy.deepcopy(weights2)
	
	weights1_new[0] = weights2[0]
	weights2_new[0] = weights1[0]
	return weights1_new, weights2_new

def mutate(weights):
	for i in range(len(weights)):
		for j in range(len(weights[i])):
			if random.random() < 0.3:
				weights[i][j] += random.uniform(-0.5,0.5)
	return weights
