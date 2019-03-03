# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 02:34:40 2019

@author: Ankit Goyal
"""
import mlrose
import numpy as np
import random 
import os
import pandas as pd
import math
import time
import mlrose
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


weights=[10,4,3,2,7,9,6,8,12,14,15,18,43,23,67,15,27,18,14,12,27,45,20]
values=[500,25,45,900,345,456,768,980,234,564,432,568,43,11,10,11,34,23,459,23,65,32,6889]
print(len(weights),len(values))

max_weight_pct=0.5
fitness= mlrose.Knapsack(weights, values, max_weight_pct)
problem_fit=mlrose.DiscreteOpt(length=19,fitness_fn=fitness,maximize=True,max_val=3)
schedule = mlrose.ExpDecay()
init_state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
fitness1= mlrose.Knapsack(weights[0:4], values[0:4], max_weight_pct)
fitness2=mlrose.Knapsack(weights[0:7], values[0:7], max_weight_pct)
fitness3=mlrose.Knapsack(weights[0:9], values[0:9], max_weight_pct)
fitness4=mlrose.Knapsack(weights[0:11], values[0:11], max_weight_pct)
fitness5=mlrose.Knapsack(weights[0:13], values[0:13], max_weight_pct)
fitness6=mlrose.Knapsack(weights[0:16], values[0:16], max_weight_pct)
fitness7=mlrose.Knapsack(weights[0:19], values[0:19], max_weight_pct)
fitness8=mlrose.Knapsack(weights[0:20], values[0:20], max_weight_pct)
fitness9=mlrose.Knapsack(weights[0:22], values[0:22], max_weight_pct)


problem_fit1=mlrose.DiscreteOpt(length =4, fitness_fn = fitness1, maximize=True,max_val=3)
problem_fit2=mlrose.DiscreteOpt(length =7, fitness_fn = fitness2, maximize=True,max_val=3)
problem_fit3=mlrose.DiscreteOpt(length =9, fitness_fn = fitness3, maximize=True,max_val=3)
problem_fit4=mlrose.DiscreteOpt(length =11, fitness_fn = fitness4, maximize=True,max_val=3)
problem_fit5=mlrose.DiscreteOpt(length =13, fitness_fn = fitness5, maximize=True,max_val=3)
problem_fit6=mlrose.DiscreteOpt(length =16, fitness_fn = fitness6, maximize=True,max_val=3)
problem_fit7=mlrose.DiscreteOpt(length =19, fitness_fn = fitness7, maximize=True,max_val=3)
problem_fit8=mlrose.DiscreteOpt(length =20, fitness_fn = fitness8, maximize=True,max_val=3)
problem_fit9=mlrose.DiscreteOpt(length =22, fitness_fn = fitness9, maximize=True,max_val=3)

np.random.seed(2)
# Solve using genetic algorithm
start=time.time()
best_state1, best_fitness1 = mlrose.genetic_alg(problem_fit1, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t1=time.time()-start
start=time.time()
best_state2, best_fitness2 = mlrose.genetic_alg(problem_fit2, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t2=time.time()-start
start=time.time()
best_state3, best_fitness3 = mlrose.genetic_alg(problem_fit3, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t3=time.time()-start
start=time.time()
best_state4, best_fitness4 = mlrose.genetic_alg(problem_fit4, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t4=time.time()-start
start=time.time()
best_state5, best_fitness5 = mlrose.genetic_alg(problem_fit5, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t5=time.time()-start
start=time.time()
best_state6, best_fitness6 = mlrose.genetic_alg(problem_fit6, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t6=time.time()-start
start=time.time()
best_state7, best_fitness7 = mlrose.genetic_alg(problem_fit7, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t7=time.time()-start
start=time.time()
best_state8, best_fitness8 = mlrose.genetic_alg(problem_fit8, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t8=time.time()-start
start=time.time()
best_state9, best_fitness9 = mlrose.genetic_alg(problem_fit9, mutation_prob = 0.7, max_attempts = 100,max_iters=1000)
t9=time.time()-start
#best_state10, best_fitness10 = mlrose.genetic_alg(problem_fit10, mutation_prob = 0.2, max_attempts = 100)

time1=[t1,t2,t3,t4,t5,t6,t7,t8,t9]
start=time.time()
best_state10, best_fitness10 = mlrose.mimic(problem_fit1,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t1=time.time()-start
start=time.time()
best_state11, best_fitness11 = mlrose.mimic(problem_fit2,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t2=time.time()-start
start=time.time()
best_state12, best_fitness12 = mlrose.mimic(problem_fit3,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t3=time.time()-start
start=time.time()
best_state13, best_fitness13 = mlrose.mimic(problem_fit4,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t4=time.time()-start
start=time.time()
best_state14, best_fitness14 = mlrose.mimic(problem_fit5,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t5=time.time()-start
start=time.time()
best_state15, best_fitness15 = mlrose.mimic(problem_fit6,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t6=time.time()-start
start=time.time()
best_state16, best_fitness16 = mlrose.mimic(problem_fit7,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t7=time.time()-start
start=time.time()
best_state17, best_fitness17 = mlrose.mimic(problem_fit8,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t8=time.time()-start
start=time.time()
best_state18, best_fitness18 = mlrose.mimic(problem_fit9,pop_size=200,keep_pct=0.7,max_attempts=10,max_iters=1000)
t9=time.time()-start
#best_state19, best_fitness19 = mlrose.mimic(problem_fit10,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)

time2=[t1,t2,t3,t4,t5,t6,t7,t8,t9]

start=time.time()
best_state19, best_fitness19 = mlrose.simulated_annealing(problem_fit1,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t1=time.time()-start
start=time.time()
best_state20, best_fitness20 = mlrose.simulated_annealing(problem_fit2,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t2=time.time()-start
start=time.time()
best_state21, best_fitness21 = mlrose.simulated_annealing(problem_fit3,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t3=time.time()-start
start=time.time()
best_state22, best_fitness22 = mlrose.simulated_annealing(problem_fit4,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t4=time.time()-start
start=time.time()
best_state23, best_fitness23 = mlrose.simulated_annealing(problem_fit5,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t5=time.time()-start
start=time.time()
best_state24, best_fitness24 = mlrose.simulated_annealing(problem_fit6,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t6=time.time()-start
start=time.time()
best_state25, best_fitness25 = mlrose.simulated_annealing(problem_fit7,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t7=time.time()-start
start=time.time()
best_state26, best_fitness26 = mlrose.simulated_annealing(problem_fit8,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t8=time.time()-start
start=time.time()
best_state27, best_fitness27 = mlrose.simulated_annealing(problem_fit9,schedule=mlrose.GeomDecay(init_temp=10000),max_attempts=100,max_iters=1000,init_state=None)
t9=time.time()-start
#start=time.time()
#best_state28, best_fitness28 = mlrose.simulated_annealing(problem_fit10,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
#t10=time.time()-start

time3=[t1,t2,t3,t4,t5,t6,t7,t8,t9]
Y1=[best_fitness1,best_fitness9,best_fitness2,best_fitness3,best_fitness4,best_fitness5,best_fitness6,best_fitness7,best_fitness8]
Y2=[best_fitness10,best_fitness18,best_fitness11,best_fitness12,best_fitness13,best_fitness14,best_fitness15,best_fitness16,best_fitness17]
Y3=[best_fitness19,best_fitness27,best_fitness20,best_fitness21,best_fitness22,best_fitness23,best_fitness24,best_fitness25,best_fitness26]
X1=[4,7,9,11,13,16,19,20,22]


line1, = plt.plot(X1,Y1,color='r',label='genetic_algorithm')
line2, = plt.plot(X1,Y2,color='g',label='mimic')
line3, = plt.plot(X1,Y3,color='b',label='simulated_annealing')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=3)})
plt.ylabel('Time (seconds)')
plt.xlabel('Number of inputs')
plt.show()


#best_state, best_fitness = mlrose.simulated_annealing(problem_fit, schedule = schedule,max_attempts = 10, max_iters =1000,init_state = init_state)
#print(best_state,best_fitness)

#best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_attempts=10,max_iters=1000,init_state=init_state)
#print(best_state,best_fitness)

#best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 100)
#print(best_state,best_fitness)

#best_state, best_fitness = mlrose.mimic(problem_fit,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
#print(best_state,best_fitness)

