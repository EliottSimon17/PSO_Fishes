import math
import numpy as np


def Rosenbrock(x, y, a=0, b=10):
    return (a - x) * (a - x) + b * (y - x * x) * (y - x * x);


def Rastrigin2D(x, y):
    return 20 + (x * x - 10 * np.cos(2 * math.pi * (x))) + (y * y - 10 * np.cos(2 * math.pi * y))


def Ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y))) - np.exp(
        0.5 * (np.cos(2 * math.pi * x) + np.cos(2 * math.pi * y))) + math.e + 20


def Sphere(x, y):
    return x * x + y * y


def Beale(x, y):
    return ((1.5 - x + (x * y)) * (1.5 - x + (x * y))) + (2.25 - x + (x * (y * y))) * (2.25 - x + (x * (y * y))) + (
            2.625 - x + x * (y * y * y)) * (2.625 - x + (x * (y * y * y)))


def Bukin(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01 * (x * x))) + 0.01 * np.abs(x + 10)


def Eggholder(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x * x + y * y)) / math.pi)))


def deriv_Rosenbrock(x, y, a=1, b=10):
    return -4 * b * x * (y - x * x) - 2 * (a - x)

def deriv_Rastrigin(x, y):

    return 20*math.pi*np.sin(2*math.pi*x) + 2*x, 20*math.pi*np.sin(2*math.pi*y) + 2*y
