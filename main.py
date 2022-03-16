# This is a sample Python script.
import matplotlib
import numpy as np
from BenchmarkFunctions import Rosenbrock, Rastrigin2D, Ackley, Sphere, Beale, Bukin, Eggholder
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
from GradientDescent import GradientDescent
from Particle import Particle, PSO
from tkinter import *

def find_optimal_value(X, Y, Z):
    opt_value = Z.min()
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            if Z[i, j] == opt_value:
                optX = X[i, j]
                optY = Y[i, j]
    return optX, optY


def plot(benchmark_func,history_values,swarmPosition,  gradient_history=None,  show_contour=True):
    if benchmark_func == 'Rosenbrock':
        # Parameters a and b
        a = 0
        b = 10
        X = np.arange(-2, 2, 0.1)
        Y = np.arange(-1, 3, 0.1)
        X, Y = np.meshgrid(X, Y)
        Z = Rosenbrock(X, Y, a, b)
        zoom_contour = 20
        zlim = 200

    elif benchmark_func == 'Rastrigin':
        X = np.arange(-4, 4, 0.01)
        Y = np.arange(-4, 4, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = Rastrigin2D(X, Y)
        zoom_contour = 10
        zlim = 100

    elif benchmark_func == 'Ackley':
        X = np.arange(-4, 4, 0.01)
        Y = np.arange(-4, 4, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = Ackley(X, Y)
        zoom_contour = 10
        zlim = 10

    elif benchmark_func == 'Sphere':
        X = np.arange(-4, 4, 0.01)
        Y = np.arange(-4, 4, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = Sphere(X, Y)
        zoom_contour = 10
        zlim = 10

    elif benchmark_func == 'Beale':
        X = np.arange(-4.4, 4.4, 0.005)
        Y = np.arange(-4.4, 4.4, 0.005)
        X, Y = np.meshgrid(X, Y)
        Z = Beale(X, Y)
        zoom_contour = 10
        zlim = 100000

    elif benchmark_func == 'Bukin':
        X = np.arange(-15, -5, 0.002)
        Y = np.arange(-2, 2, 0.002)
        X, Y = np.meshgrid(X, Y)
        Z = Bukin(X, Y)
        zoom_contour = 10
        zlim = 200
    elif benchmark_func == 'Eggholder':
        X = np.arange(-10, 10, 0.01)
        Y = np.arange(-10, 10,  0.01)
        X, Y = np.meshgrid(X, Y)
        Z = Eggholder(X, Y)
        zoom_contour = 1
        zlim = 5
    figRos = plt.figure(figsize=(12, 7))
    axRos = figRos.gca(projection='3d')

    surf = axRos.plot_surface(X, Y, Z, cmap=cm.hsv, alpha=.5)
    if benchmark_func == 'Eggholder':
        axRos.set_zlim(-18, 0)
    else:
        axRos.set_zlim(0, zlim)
    figRos.colorbar(surf, shrink=0.5)

    # Plot the true optimal point
    optX, optY = find_optimal_value(X, Y, Z)
    print('True Optimum Function: ')
    print('X: ', optX)
    print('Y: ', optY)
    plt.plot(optX, optY, 'x', color='Green', markersize=5, label='Optimal Point')
    # Print all the PSO steps
    for i in range(len(history_values)):
        plt.plot(history_values[i][0], history_values[i][1], 'x', color='Black', markersize=3)
    # Plots Gradient Descent Steps
    if gradient_history is not None:
        for i in range(len(gradient_history)):
            if i % 10 == 0 or i < 10:
                plt.plot(gradient_history[i][0], gradient_history[i][1], 'x', color='Red', markersize=3)
    plt.show()

    if show_contour:
        if swarmPosition is None:
            fig, ax = plt.subplots(1, 1)
            ax.plot(optX, optY, 'x', color='Green', markersize=5,
                    label='Optimal Point')
            ax.contour(X, Y, Z, zoom_contour)
            ax.plot(history_values[0][0], history_values[0][1], 'x', color='Black', markersize=5, label='PSO iterations')
            if gradient_history is not None:
                ax.plot(gradient_history[0][0], gradient_history[0][1], 'x', color='Red', markersize=5,  label='GD iterations')


            for i in range(1,len(history_values)):
                ax.plot(history_values[i][0], history_values[i][1], 'x', color='Black', markersize = 3)


            if gradient_history is not None:
                for i in range(1,len(gradient_history)):
                    if i % 10 == 0 or i < 10:
                        ax.plot(gradient_history[i][0], gradient_history[i][1], 'x', color='Red', markersize=3)
            ax.legend()
            plt.show()
        else:
            index = 0
            for swarm in swarmPosition:
                index = index +1
                if index % 5==0 or index == 0:
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(optX, optY, 'x', color='Green', markersize=5,
                            label='Optimal Point')
                    ax.contour(X, Y, Z, zoom_contour)
                    ax.plot(history_values[0][0], history_values[0][1], 'x', color='Black', markersize=5,
                            label='PSO iterations')
                    if gradient_history is not None:
                        ax.plot(gradient_history[0][0], gradient_history[0][1], 'x', color='Red', markersize=5,
                                label='GD iterations')

                    for particle in swarm:
                            ax.plot(particle[0], particle[1], 'x', color='purple', markersize=5)

                    if gradient_history is not None:
                        for i in range(1, len(gradient_history)):
                            if i % 10 == 0 or i < 10:
                                ax.plot(gradient_history[i][0], gradient_history[i][1], 'x', color='Red', markersize=3)
                    ax.legend()
                    plt.show()
    return X, Y, Z


def action(function):
    p = PSO()
    points_iterations, swarm_positions = p.launch(function)

    if function == 'Rosenbrock' or function == 'Rastrigin':
        gd = GradientDescent(function)
        gradient_history = gd.iterate()
        print(gradient_history)
        gradient_XY = []
        for i in range(len(gradient_history)):
            gradient_XY.append(gradient_history[i][0])
        X, Y, z = plot(function, points_iterations,swarm_positions , gradient_XY)
    else:
        X, Y, z = plot(function, points_iterations, swarmPosition=None)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    window = Tk()
    window.geometry('500x200')
    benchmark_function = ['Rosenbrock', 'Rastrigin', 'Ackley', 'Sphere', 'Beale', 'Bukin', 'Eggholder']

    btn1 = Button(window, text='Rosenbrock',command= lambda: action(btn1['text']))
    btn2 = Button(window, text='Rastrigin', command=lambda: action(btn2['text']))
    btn3 = Button(window, text='Ackley', command= lambda:action(btn3['text']))
    btn4 = Button(window, text='Sphere', command=lambda: action(btn4['text']))
    btn5 = Button(window, text='Beale', command=lambda: action(btn5['text']))
    btn6 = Button(window, text='Bukin', command=lambda: action(btn6['text']))
    btn7 = Button(window, text='Eggholder', command=lambda: action(btn7['text']))
    btn1.pack(side='top')
    btn2.pack(side='top')
    btn3.pack(side='top')
    btn4.pack(side='top')
    btn5.pack(side='top')
    btn6.pack(side='top')
    btn7.pack(side='top')

    window.mainloop()