import os

import numpy as np
import random

class World:

    def __init__(self, width=10, height=10):

        self.width = width
        self.height = height
        self.map = None
        self.generate_world()
        self.make_food(4)

        self.player_loc = [6, 5]

    def generate_world(self):

        self.map = [["-" for row in range(self.width)] for col in range(self.height)]

    def make_food(self, num):

        for food in range(num):
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            while self.map[x][y] == "O":
                x = random.randint(0, self.width-1)
                y = random.randint(0, self.height-1)

            self.map[x][y] = "O"

    def update_player(self, player):

        self.map[self.player_loc[0]][self.player_loc[1]] = "-"
        if self.map[player.loc[0]][player.loc[1]] == "O":
            player.food += 1
            player.reward = 1.0
        else:
            player.reward = 0.2
        self.map[player.loc[0]][player.loc[1]] = "P"
        self.player_loc[0] = player.loc[0]
        self.player_loc[1] = player.loc[1]
        # os.system('cls')
        # print(np.array(self.map))
        # print("Hunger: ", player.hunger)
        # print("HP: ", player.health)
        # print("Food: ", player.food)




if __name__ == '__main__':
    w1 = World()
    print(np.array(w1.map))
