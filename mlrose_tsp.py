# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:10:33 2019

@author: Ankit Goyal
"""

import random 
import os
import numpy as np
import pandas as pd
import math
import time
import mlrose
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D



os.getcwd()
os.chdir('C:/Users/Ankit Goyal/Desktop/OMSCS program/Pre-Reqs_Data_Science/Course ML/Assignment 2')
city=pd.read_csv('Test.csv')
#city=pd.read.csv("Test.csv")


coords3=[(1,54),(4,4),(5,324),(6,123),(7,23),(9,214),(10,267),(34,3),(56,32),(101,39),(345,76),(45,34),(45,67),(34,76),(12,43),(56,27),(89,12),(15,43),(234,65),(58,38),(97,32)]

fitness_coords1 = mlrose.TravellingSales(coords = coords3[0:3])
fitness_coords2 = mlrose.TravellingSales(coords = coords3[0:7])
fitness_coords3 = mlrose.TravellingSales(coords = coords3[0:10])
fitness_coords4 = mlrose.TravellingSales(coords = coords3[0:12])
fitness_coords5 = mlrose.TravellingSales(coords = coords3[0:16])
fitness_coords6 = mlrose.TravellingSales(coords = coords3[0:17])
fitness_coords7 = mlrose.TravellingSales(coords = coords3[0:18])
fitness_coords8 = mlrose.TravellingSales(coords = coords3[0:19])
fitness_coords9 = mlrose.TravellingSales(coords = coords3[0:5])
fitness_coords10 = mlrose.TravellingSales(coords = coords3[0:20])


problem_fit1=mlrose.TSPOpt(length =3, fitness_fn = fitness_coords1, maximize=False)
problem_fit2=mlrose.TSPOpt(length =7, fitness_fn = fitness_coords2, maximize=False)
problem_fit3=mlrose.TSPOpt(length =10, fitness_fn = fitness_coords3, maximize=False)
problem_fit4=mlrose.TSPOpt(length =12, fitness_fn = fitness_coords4, maximize=False)
problem_fit5=mlrose.TSPOpt(length =16, fitness_fn = fitness_coords5, maximize=False)
problem_fit6=mlrose.TSPOpt(length =17, fitness_fn = fitness_coords6, maximize=False)
problem_fit7=mlrose.TSPOpt(length =18, fitness_fn = fitness_coords7, maximize=False)
problem_fit8=mlrose.TSPOpt(length =19, fitness_fn = fitness_coords8, maximize=False)
problem_fit9=mlrose.TSPOpt(length =5, fitness_fn = fitness_coords9, maximize=False)
problem_fit10=mlrose.TSPOpt(length =20, fitness_fn = fitness_coords10, maximize=False)

np.random.seed(2)
# Solve using genetic algorithm

start=time.time()
best_state1, best_fitness1 = mlrose.genetic_alg(problem_fit1, mutation_prob = 0.2, max_attempts = 100,max_iters=1000)
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
t6=time.time()
start=time.time()
best_state7, best_fitness7 = mlrose.genetic_alg(problem_fit7, mutation_prob = 0.2, max_attempts = 100)
t7=time.time()-start
start=time.time()
best_state8, best_fitness8 = mlrose.genetic_alg(problem_fit8, mutation_prob = 0.2, max_attempts = 100)
t8=time.time()-start
start=time.time()
best_state9, best_fitness9 = mlrose.genetic_alg(problem_fit9, mutation_prob = 0.2, max_attempts = 100)
t9=time.time()-start
#best_state10, best_fitness10 = mlrose.genetic_alg(problem_fit10, mutation_prob = 0.2, max_attempts = 100)

time1=[t1,t2,t3,t4,t5]

start=time.time()
best_state10, best_fitness10 = mlrose.mimic(problem_fit1,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t1=time.time()-start
start=time.time()
best_state11, best_fitness11 = mlrose.mimic(problem_fit2,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t2=time.time()-start
start=time.time()
best_state12, best_fitness12 = mlrose.mimic(problem_fit3,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t3=time.time()-start
start=time.time()
best_state13, best_fitness13 = mlrose.mimic(problem_fit4,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t4=time.time()-start
start=time.time()
best_state14, best_fitness14 = mlrose.mimic(problem_fit5,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t5=time.time()-start
start=time.time()
best_state15, best_fitness15 = mlrose.mimic(problem_fit6,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t6=time.time()-start
start=time.time()
best_state16, best_fitness16 = mlrose.mimic(problem_fit7,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t7=time.time()-start
start=time.time()
best_state17, best_fitness17 = mlrose.mimic(problem_fit8,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t8=time.time()-start
start=time.time()
best_state18, best_fitness18 = mlrose.mimic(problem_fit9,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
t9=time.time()-start
#best_state19, best_fitness19 = mlrose.mimic(problem_fit10,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)

time2=[t1,t2,t3,t4,t5]

start=time.time()
best_state19, best_fitness19 = mlrose.simulated_annealing(problem_fit1,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t1=time.time()-start
start=time.time()
best_state20, best_fitness20 = mlrose.simulated_annealing(problem_fit2,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t2=time.time()-start
start=time.time()
best_state21, best_fitness21 = mlrose.simulated_annealing(problem_fit3,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t3=time.time()-start
start=time.time()
best_state22, best_fitness22 = mlrose.simulated_annealing(problem_fit4,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t4=time.time()-start
start=time.time()
best_state23, best_fitness23 = mlrose.simulated_annealing(problem_fit5,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t5=time.time()-start
start=time.time()
best_state24, best_fitness24 = mlrose.simulated_annealing(problem_fit6,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t6=time.time()-start
start=time.time()
best_state25, best_fitness25 = mlrose.simulated_annealing(problem_fit7,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t7=time.time()-start
start=time.time()
best_state26, best_fitness26 = mlrose.simulated_annealing(problem_fit8,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t8=time.time()-start
start=time.time()
best_state27, best_fitness27 = mlrose.simulated_annealing(problem_fit9,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
t9=time.time()-start
#best_state28, best_fitness28 = mlrose.simulated_annealing(problem_fit10,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)
time3=[t1,t2,t3,t4,t5]

Y1=[best_fitness1,best_fitness9,best_fitness2,best_fitness3,best_fitness4,best_fitness5,best_fitness6,best_fitness7,best_fitness8]
Y2=[best_fitness10,best_fitness18,best_fitness11,best_fitness12,best_fitness13,best_fitness14,best_fitness15,best_fitness16,best_fitness17]
Y3=[best_fitness19,best_fitness27,best_fitness20,best_fitness21,best_fitness22,best_fitness23,best_fitness24,best_fitness25,best_fitness26]
X1=[3,5,7,10,12]


line1, = plt.plot(X1,time1,color='r',label='genetic_algorithm')
line2, = plt.plot(X1,time2,color='g',label='mimic')
line3, = plt.plot(X1,time3,color='b',label='simulated_annealing')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=3)})
plt.ylabel('Time(seconds)')
plt.xlabel('Number of inputs')
plt.show()

#print(best_state)
#print(best_fitness)
#Solve using Random Hill Climbing
#best_state, best_fitness = mlrose.random_hill_climb(problem_fit, max_attempts=20,max_iters=1000,init_state=None)
#print(best_state)
#print(best_fitness)
#best_state, best_fitness = mlrose.mimic(problem_fit,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=1000)
#print(best_state)
#print(best_fitness)
#best_state, best_fitness = mlrose.simulated_annealing(problem_fit,schedule=mlrose.GeomDecay(),max_attempts=100,max_iters=1000,init_state=None)

#print(best_state)
#print(best_fitness)

def max_iterations(n,cords,fit,temp,mint):
    best_fit1=[]
    best_fit2=[]
    best_fit3=[]
    max_iter=[]
    for i in range(100,n,10):
        best_state,best_fitness1 = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 100,max_iters=i)
        best_state,best_fitness2 = mlrose.mimic(problem_fit,pop_size=200,keep_pct=0.3,max_attempts=10,max_iters=i)
        best_state,best_fitness3 = mlrose.simulated_annealing(problem_fit,schedule=mlrose.GeomDecay(init_temp=temp,decay=0.9,min_temp=mint),max_attempts=100,max_iters=i,init_state=None)
        max_iter.append(i)
        best_fit1.append(best_fitness1)
        best_fit2.append(best_fitness2)
        best_fit3.append(best_fitness3)
        print(best_fit1,best_fit2,best_fit3)
    max_iter=np.asarray(max_iter)
    best_fit1=np.asarray(best_fit1)
    best_fit2=np.asarray(best_fit2)
    best_fit3=np.asarray(best_fit3)
    line1, = plt.plot(max_iter,best_fit1,color='r',label='fitness_score')
    line2, = plt.plot(max_iter,best_fit2,color='g',label='fitness_score')
    line3, = plt.plot(max_iter,best_fit3,color='b',label='fitness_score')
    plt.ylabel('Fitness_score')
    plt.xlabel('Number of iterations')
    plt.show()
    return None

#max_iterations(200,fitness_coords,problem_fit,1000,0.1)

def mutation_probab(max_probab,cords,fit):
    best_fit=[]
    mutation_prob=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for i in mutation_prob:
        best_state, best_fitness = mlrose.genetic_alg(problem_fit, mutation_prob = 0.2, max_attempts = 100)
        #max_iter.append(i)
        best_fit.append(best_fitness)
        print(best_fit)
    #max_iter=np.asarray(mutation_probab)
    #mutation_prob=np.asarray(mutation_prob)
    best_fit=np.asarray(best_fit)
    print(best_fit)
    line1, = plt.plot(mutation_prob,best_fit,color='r',label='fitness_score')
    plt.ylabel('Fitness_score')
    plt.xlabel('Mutation Probability')
    plt.show()
    return None

#mutation_probab(1,fitness_coords,problem_fit)       

def max_attempts_mimic(n,cords,fit):
    best_fit=[]
    max_iter=[]
    for i in range(10,n):
        best_state, best_fitness = mlrose.mimic(problem_fit,pop_size=200,keep_pct=0.3,max_attempts=i,max_iters=1000)
        max_iter.append(i)
        best_fit.append(best_fitness)
        print(best_fit)
    max_iter=np.asarray(max_iter)
    best_fit=np.asarray(best_fit)
    line1, = plt.plot(max_iter,best_fit,color='r',label='fitness_score')
    plt.ylabel('Fitness_score')
    plt.xlabel('Number of attempts')
    plt.show()
    return None
#max_attempts_mimic(100,fitness_coords,problem_fit)

def pct_mimic(cords,fit):
    best_fit=[]
    mutation_prob=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for i in mutation_prob:
        best_state, best_fitness = mlrose.mimic(problem_fit,pop_size=200,keep_pct=i,max_attempts=10,max_iters=1000)
        #max_iter.append(i)
        best_fit.append(best_fitness)
        print(best_fit)
    #max_iter=np.asarray(mutation_probab)
    #mutation_prob=np.asarray(mutation_prob)
    best_fit=np.asarray(best_fit)
    print(best_fit)
    line1, = plt.plot(mutation_prob,best_fit,color='r',label='fitness_score')
    plt.ylabel('Fitness_score')
    plt.xlabel('pct')
    plt.show()
    return None

#pct_mimic(fitness_coords,problem_fit)

def temp_SA(cords,fit,temp):
    best_fit=[]
    max_iter=[]
    for i in range(10,temp,1):
        best_state,best_fitness3 = mlrose.simulated_annealing(problem_fit,schedule=mlrose.GeomDecay(init_temp=i,decay=0.9,min_temp=0.1),max_attempts=100,max_iters=100,init_state=None)
        max_iter.append(i)
        best_fit.append(best_fitness3)
        print(best_fit)
    max_iter=np.asarray(max_iter)
    best_fit=np.asarray(best_fit)
    line1, = plt.plot(max_iter,best_fit,color='r',label='fitness_score')
    plt.ylabel('Fitness_score')
    plt.xlabel('Number of attempts')
    plt.show()
    return None
