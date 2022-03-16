import numpy as np
from random import random
from random import uniform
from BenchmarkFunctions import Rosenbrock, Rastrigin2D, Ackley, Sphere, Beale, Bukin, Eggholder


class Particle:

    def __init__(self, dimensions, vmax=1):
        # generate random velocity [-vmax, vmax]
        self.velocity = 2 * vmax * np.random.rand(dimensions, 1) - vmax
        # Spawn the particles at random locations
        r1 = uniform(-3, 3)
        r2 = uniform(-3, 3)
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
    def __init__(self, population_size=20, dimensions=2, alpha=0.9, b=2, c=2, vmax=1, pos_init=None):
        self.population_size = population_size
        self.dimensions = dimensions
        self.alpha = alpha
        self.b = b
        self.c = c
        self.vmax = vmax
        self.pos_init = pos_init
    # Initializes population, given benchmark function
    def generatePopulation(self, benchmark_function):
        self.population = []
        for i in range(self.population_size):
            self.population.append(Particle(dimensions=self.dimensions, vmax= self.vmax))
            if i == 0:
                self.global_best = self.population[0].pos.copy()
            else:
                self.global_best = self.evaluate(self.population[i].pos, self.global_best, benchmark_function)

    # This function applies the function that is specified to compute the utility
    def compute_z(self, benchmark_function, pos1, pos2):
        if benchmark_function == 'Rosenbrock':
            return Rosenbrock(pos1 , pos2)
        elif benchmark_function == 'Rastrigin':
            return Rastrigin2D(pos1 , pos2)
        elif benchmark_function == 'Ackley':
            return Ackley(pos1,pos2)
        elif benchmark_function ==  'Sphere':
            return Sphere(pos1,pos2)
        elif benchmark_function == 'Beale':
            return Beale(pos1,pos2)
        elif benchmark_function == 'Bukin':
            return Bukin(pos1,pos2)
        elif benchmark_function == 'Eggholder':
            return Eggholder(pos1,pos2)

    def evaluate(self, position , current_best, benchmark_function):
        if self.compute_z(benchmark_function, position[0], position[1]) < self.compute_z(benchmark_function, current_best[0], current_best[1]):
            current_best = position.copy()

        return current_best

    def iterate(self, benchmark_function):
        self.t = 0
        self.fitness_time, self.time = [], []

        number_iteration = 400
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
                particle.pbest = self.evaluate(particle.pos , particle.pbest, benchmark_function)
                self.global_best = self.evaluate(particle.pos , self.global_best, benchmark_function)
                self.fitness_time.append(self.compute_z(benchmark_function, self.global_best[0], self.global_best))
                self.time.append(self.t)
            print('Iteration number : ', self.t, ', best global cost:',self.compute_z(benchmark_function, self.global_best[0], self.global_best[1]))
            position_history.append(position_swarm)
            self.t += 1
            history_scores.append(self.global_best)
        return history_scores, position_history

    def launch(self, benchmark_function):

        self.generatePopulation(benchmark_function)
        history_scores, position_swarm = self.iterate(benchmark_function)
        select_for_plot = []
        for i in range(len(history_scores)):
            select_for_plot.append(history_scores[i])
        print('\nOptimal Solution (X,Y)  >', np.round(self.global_best.reshape(-1), 7).tolist())
        print('\nOptimal Z-value (PSO) :>', self.compute_z(benchmark_function,self.global_best[0], self.global_best[1]), '\n')
        return select_for_plot, position_swarm