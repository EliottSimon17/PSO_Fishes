import numpy as np
from random import random
from random import uniform
from BenchmarkFunctions import Rosenbrock, Rastrigin2D, Ackley, Sphere, Beale, Bukin, Eggholder


class Bird:

    def __init__(self, dimensions, vmax=1):
        # generate random velocity [-vmax, vmax]
        self.velocity = 2 * vmax * np.random.rand(dimensions, 1) - vmax
        # Spawn the particles at random locations
        r1 = uniform(0, 1000)
        r2 = uniform(0, 900)
        self.pos = [[r1], [r2]]
        self.pos = np.array(self.pos)
        self.best_position = []
        self.pbest = self.pos.copy()
        self.vmax = vmax

    def update_velocity(self, alpha, b, c, global_best):
        # Formula for updating the velocity
        R = np.random.rand()
        self.velocity = self.velocity * alpha
        self.velocity += b * R * (self.pbest - self.pos)
        self.velocity += c * R * (global_best - self.pos)
        self.velocity = np.clip(self.velocity, -self.vmax, self.vmax)

    # Euler integration to update position
    def update_position(self):
        self.pos = self.pos + self.velocity


class PSO:
    def __init__(self, population_size=50, dimensions=2, alpha=0.001, b=5, c=5, vmax=5, pos_init=None):
        self.population_size = population_size
        self.dimensions = dimensions
        self.alpha = alpha
        self.b = b
        self.c = c
        self.vmax = vmax
        self.pos_init = pos_init
    # Initializes population, given benchmark function
    def generatePopulation(self, fishes):
        self.population = []
        for i in range(self.population_size):
            self.population.append(Bird(dimensions=self.dimensions, vmax= self.vmax))
            if i == 0:
                self.global_best = self.population[0].pos.copy()
            else:
                self.global_best = self.evaluate(self.population[i].pos, self.global_best, fishes)


    def count_bird(self, positionX, positionY, fishes):
        count = 0
        for fish in fishes:
            if ((fish[0]-positionX)**2 + (fish[1]-positionY)**2 ) < (30*30):
                count = count+1
        return count

    def evaluate(self, position , current_best, fishes):
        if self.count_bird(position[0], position[1], fishes) > self.count_bird(current_best[0], current_best[1], fishes):
            current_best = position.copy()
        return current_best

    def iterate(self, fishes):
        self.t = 0
        self.fitness_time, self.time = [], []

        number_iteration = 100
        history_scores = []
        position_history = []
        while self.t <= number_iteration:
            if self.alpha >= 0.01:
                self.alpha = self.alpha - 0.001
            position_swarm = []
            for particle in self.population:
                position_swarm.append(particle.pos)
                particle.update_velocity(self.alpha, self.b, self.c, self.global_best)
                particle.update_position()
                particle.pbest = self.evaluate(particle.pos , particle.pbest, fishes)
                self.global_best = self.evaluate(particle.pos , self.global_best, fishes)
                self.time.append(self.t)
            print('Iteration number : ', self.t, ', best global fitness:',self.count_bird( self.global_best[0], self.global_best[1], fishes))
            position_history.append(position_swarm)
            self.t += 1
            history_scores.append(self.global_best)
        return history_scores, position_history

    def launch(self, fishes):

        self.generatePopulation(fishes)
        history_scores, position_swarm = self.iterate(fishes)
        select_for_plot = []
        for i in range(len(history_scores)):
            select_for_plot.append(history_scores[i])
        print('\nOptimal Solution (X,Y)  >', np.round(self.global_best.reshape(-1), 7).tolist())
        #print('\nOptimal Z-value (PSO) :>', self.compute_z(fishes,self.global_best[0], self.global_best[1]), '\n')
        return select_for_plot, position_swarm