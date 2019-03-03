# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:43:08 2019

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

edges=[(0,1),(0,2),(0,4),(1,3),(2,0),(2,3),(3,4),(5,6),(6,8),(7,6),(3,6),(6,8),(1,9),(3,9),(6,9),(7,9),(1,10),(6,10),(3,9),(11,5),(6,11),(1,11),(5,7),(3,11),(2,10),(7,11),(6,12),(13,8),(14,5),(13,12),(15,4),(12,16),(17,18),(18,19),(19,2),(20,4),(6,19),(11,21),(22,15),(10,23)]
schedule = mlrose.ExpDecay()


fitness1= mlrose.MaxKColor(edges[0:5])
fitness2=mlrose.MaxKColor(edges[0:7])
fitness3=mlrose.MaxKColor(edges[0:9])
fitness4=mlrose.MaxKColor(edges[0:11])
fitness5=mlrose.MaxKColor(edges[0:13])
fitness6=mlrose.MaxKColor(edges[0:15])
fitness7=mlrose.MaxKColor(edges[0:19])
fitness8=mlrose.MaxKColor(edges[0:21])
fitness9=mlrose.MaxKColor(edges[0:24])
fitness10=mlrose.MaxKColor(edges[0:30])

start=time.time()
problem_fit1=mlrose.DiscreteOpt(length =5, fitness_fn = fitness1, maximize=True,max_val=4)
t1=time.time()-start
start=time.time()
problem_fit2=mlrose.DiscreteOpt(length =7, fitness_fn = fitness2, maximize=True,max_val=4)
t2=time.time()-start
start=time.time()
problem_fit3=mlrose.DiscreteOpt(length =9, fitness_fn = fitness3, maximize=True,max_val=4)
t3=time.time()-start
start=time.time()
problem_fit4=mlrose.DiscreteOpt(length =11, fitness_fn = fitness4, maximize=True,max_val=4)
t4=time.time()-start
start=time.time()
problem_fit5=mlrose.DiscreteOpt(length =13, fitness_fn = fitness5, maximize=True,max_val=4)
t5=time.time()-start
start=time.time()
problem_fit6=mlrose.DiscreteOpt(length =15, fitness_fn = fitness6, maximize=True,max_val=4)
t6=time.time()-start
start=time.time()
problem_fit7=mlrose.DiscreteOpt(length =19, fitness_fn = fitness7, maximize=True,max_val=4)
t7=time.time()-start
start=time.time()
problem_fit8=mlrose.DiscreteOpt(length =21, fitness_fn = fitness8, maximize=True,max_val=4)
t8=time.time()-start
start=time.time()
problem_fit9=mlrose.DiscreteOpt(length =24, fitness_fn = fitness9, maximize=True,max_val=4)
t9=time.time()-start
start=time.time()
problem_fit10=mlrose.DiscreteOpt(length =30, fitness_fn = fitness10, maximize=True,max_val=4)
t10=time.time()-start



np.random.seed(2)
# Solve using genetic algorithm
start=time.time()
best_state1, best_fitness1 = mlrose.genetic_alg(problem_fit1, mutation_prob = 0.2, max_attempts = 100)
t1=time.time()-start
start=time.time()
best_state2, best_fitness2 = mlrose.genetic_alg(problem_fit2, mutation_prob = 0.2, max_attempts = 100)
t2=time.time()-start
start=time.time()
best_state3, best_fitness3 = mlrose.genetic_alg(problem_fit3, mutation_prob = 0.2, max_attempts = 100)
t3=time.time()-start
start=time.time()
best_state4, best_fitness4 = mlrose.genetic_alg(problem_fit4, mutation_prob = 0.2, max_attempts = 100)
t4=time.time()-start
start=time.time()
best_state5, best_fitness5 = mlrose.genetic_alg(problem_fit5, mutation_prob = 0.2, max_attempts = 100)
t5=time.time()-start
start=time.time()
best_state6, best_fitness6 = mlrose.genetic_alg(problem_fit6, mutation_prob = 0.2, max_attempts = 100)
t6=time.time()-start
start=time.time()
best_state7, best_fitness7 = mlrose.genetic_alg(problem_fit7, mutation_prob = 0.2, max_attempts = 100)
t7=time.time()-start
start=time.time()
best_state8, best_fitness8 = mlrose.genetic_alg(problem_fit8, mutation_prob = 0.2, max_attempts = 100)
t8=time.time()-start
start=time.time()
best_state9, best_fitness9 = mlrose.genetic_alg(problem_fit9, mutation_prob = 0.2, max_attempts = 100)
t9=time.time()-start
start=time.time()
best_state29, best_fitness29 = mlrose.genetic_alg(problem_fit10, mutation_prob = 0.2, max_attempts = 100)
t10=time.time()-start

time1=[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10]
#best_state10, best_fitness10 = mlrose.genetic_alg(problem_fit10, mutation_prob = 0.2, max_attempts = 100)


start=time.time()
best_state10, best_fitness10 = mlrose.mimic(problem_fit1,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t21=time.time()-start
start=time.time()
best_state11, best_fitness11 = mlrose.mimic(problem_fit2,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t22=time.time()-start
start=time.time()
best_state12, best_fitness12 = mlrose.mimic(problem_fit3,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t23=time.time()-start
start=time.time()
best_state13, best_fitness13 = mlrose.mimic(problem_fit4,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t24=time.time()-start
start=time.time()
best_state14, best_fitness14 = mlrose.mimic(problem_fit5,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t25=time.time()-start
start=time.time()
best_state15, best_fitness15 = mlrose.mimic(problem_fit6,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t26=time.time()-start
start=time.time()
best_state16, best_fitness16 = mlrose.mimic(problem_fit7,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t27=time.time()-start
start=time.time()
best_state17, best_fitness17 = mlrose.mimic(problem_fit8,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t28=time.time()-start
start=time.time()
best_state18, best_fitness18 = mlrose.mimic(problem_fit9,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t29=time.time()-start
start=time.time()
best_state30, best_fitness30 = mlrose.mimic(problem_fit10,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t30=time.time()-start
time2=[t21,t22,t23,t24,t25,t26,t27,t28,t29,t30]


start=time.time()
best_state19, best_fitness19 = mlrose.simulated_annealing(problem_fit1,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t11=time.time()-start
start=time.time()
best_state20, best_fitness20 = mlrose.simulated_annealing(problem_fit2,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t12=time.time()-start
start=time.time()
best_state21, best_fitness21 = mlrose.simulated_annealing(problem_fit3,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t13=time.time()-start
start=time.time()
best_state22, best_fitness22 = mlrose.simulated_annealing(problem_fit4,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t14=time.time()-start
start=time.time()
best_state23, best_fitness23 = mlrose.simulated_annealing(problem_fit5,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t15=time.time()-start
start=time.time()
best_state24, best_fitness24 = mlrose.simulated_annealing(problem_fit6,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t16=time.time()-start
start=time.time()
best_state25, best_fitness25 = mlrose.simulated_annealing(problem_fit7,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t17=time.time()-start
start=time.time()
best_state26, best_fitness26 = mlrose.simulated_annealing(problem_fit8,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t18=time.time()-start
start=time.time()
best_state27, best_fitness27 = mlrose.simulated_annealing(problem_fit9,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t19=time.time()-start
start=time.time()
best_state31, best_fitness31 = mlrose.simulated_annealing(problem_fit10,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t20=time.time()-start
time3=[t11,t12,t13,t14,t15,t16,t17,t18,t19,t20]


Y1=[best_fitness1,best_fitness2,best_fitness3,best_fitness4,best_fitness5,best_fitness6,best_fitness7,best_fitness8,best_fitness9,best_fitness29]
Y2=[best_fitness10,best_fitness11,best_fitness12,best_fitness13,best_fitness14,best_fitness15,best_fitness16,best_fitness17,best_fitness18,best_fitness30]
Y3=[best_fitness19,best_fitness20,best_fitness21,best_fitness22,best_fitness23,best_fitness24,best_fitness25,best_fitness26,best_fitness27,best_fitness31]
X1=[5,7,9,11,13,15,19,21,24,26]

line1, = plt.plot(X1,Y1,color='r',label='genetic_algorithm')
line2, = plt.plot(X1,Y2,color='g',label='mimic')
line3, = plt.plot(X1,Y3,color='b',label='simulated_annealing')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=3)})
plt.ylabel('Fitness_score')
plt.xlabel('Number of inputs')
plt.show()