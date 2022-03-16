import pygame
import numpy as np
from Bird import PSO


def generate_fishes():
    fishes = []

    central_point1 = [100, 100]
    central_point2 = [700, 500]
    central_point3 = [1000, 200]
    central_point4 = [1100, 700]
    central_point5 = [200, 600]

    central_swarm = [central_point1, central_point2, central_point3, central_point4, central_point5]

    fishes_amount = [150, 100, 130, 40, 30]

    for i in range(len(central_swarm)):
        for n in range(fishes_amount[i]):
            x = np.random.uniform(central_swarm[i][0] - fishes_amount[i], central_swarm[i][0] + fishes_amount[i])
            y = np.random.uniform(central_swarm[i][1] - fishes_amount[i], central_swarm[i][1] + fishes_amount[i])
            direction = np.random.uniform(0,np.pi*2)
            point = [x, y, direction]
            fishes.append(point)

    return fishes

if __name__ == '__main__':
    pygame.init()
    counter = 0
    # Set up the drawing window
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 900
    screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

    # Run until the user asks to quit
    running = True
    fish_army = generate_fishes()
    fish_img = pygame.image.load('fish2.png')
    fish_img = pygame.transform.scale(fish_img, (20, 20))

    p = PSO()
    points_iterations, swarm_positions = p.launch(fish_army)
    pos = []
    x_button, y_button = 550, 50
    index = 0
    swarm_points = []

    while running:
        counter = counter + 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                if x_button <= pos[0] <= x_button + 100 and y_button <= pos[1] < y_button + 50:
                    if index < len(swarm_positions):
                        swarm_points = swarm_positions[index]
                        index = index + 1

        # Fill the background with white
        screen.fill((153, 204, 255))

        for fish in fish_army:
            if fish[0] == 0:
                fish[2] = fish[2] + np.pi
            if fish[1] == 0:
                fish[2] = fish[2] + np.pi
            #if counter % 5 == 0:
                #fish[0] = fish[0] + np.cos(fish[2])
                #fish[1] = fish[1] - np.sin(fish[2])
            # pygame.draw.circle(screen, (0, 0, 255), (fish[0], fish[1]), 2)
            screen.blit(fish_img, (fish[0], fish[1]))


        if len(swarm_points)>0:
            for swarm in swarm_points:
                pygame.draw.circle(screen, (255, 0, 0), (int(swarm[0]), int(swarm[1])), 2)

        pygame.draw.rect(screen, (0, 220, 255), (x_button, y_button, 100, 50))
        screen.blit(pygame.font.SysFont('Sherif', 25).render('Iterate!', True, (0, 0, 0)), (570, 70))

       # for i in points_iterations:
       #     pygame.draw.circle(screen, (255, 0, 0), (int(i[0]), int(i[1])), 2)
        pygame.draw.circle(screen, (255, 0, 0), (
        int(points_iterations[len(points_iterations) - 1][0]), int(points_iterations[len(points_iterations) - 1][1])),
                           5)

        # Flip the display
        pygame.display.flip()

    # Done! Time to quit.
    pygame.quit()
