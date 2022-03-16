from BenchmarkFunctions import Rosenbrock, Rastrigin2D, deriv_Rosenbrock, deriv_Rastrigin
from random import uniform
import numpy as np


class GradientDescent:
    def __init__(self, benchmark_function, learning_rate=0.001, initial_position=[0, 0], iterations=400):
        # spawn particles at random location
        r1 = uniform(-3, 3)
        r2 = uniform(-3, 3)
        self.benchmark_function = benchmark_function
        self.learning_rate = learning_rate
        self.initial_position = [r1, r2]
        self.iterations = iterations

    def iterate(self):
        history = []
        for i in range(self.iterations):
            if self.benchmark_function == 'Rosenbrock':
                z_score = Rosenbrock(self.initial_position[0], self.initial_position[1])
                history.append([self.initial_position, z_score])
                derivative_z = deriv_Rosenbrock(self.initial_position[0], self.initial_position[1])
                # Move towards minimum
                self.initial_position = self.initial_position - np.dot(self.learning_rate, derivative_z)
            elif self.benchmark_function == 'Rastrigin':
                z_score = Rastrigin2D(self.initial_position[0], self.initial_position[1])
                history.append([self.initial_position, z_score])
                derivative_z_x, derivative_z_y = deriv_Rastrigin(self.initial_position[0], self.initial_position[1])
                # Move towards minimum
                X_change = self.initial_position[0] - np.dot(self.learning_rate, derivative_z_x)
                Y_change = self.initial_position[1] - np.dot(self.learning_rate, derivative_z_y)
                self.initial_position = [X_change, Y_change]
        print('optimal point at (X,Y) =  (', self.initial_position[0], ',', self.initial_position[1], '), with value =',
              history[-1][1])
        return history
