import time

import numpy as np

from world import World


actions = {0:"UP",
           1:"DOWN",
           2:"LEFT",
           3:"RIGHT",
           4:"EAT"}

class Player:

    def __init__(self):

        self.world = World()

        self.health = 100
        self.hunger = 100
        self.loc = [6, 5]
        self.food = 0
        self.actions = 0
        self.neighbors = []

        self.reward = 0

    def step(self, action):

        self.reward = 0
        self.scan()
        self.next_day()

        if action < 4:
            self.move(actions[action])
        else:
            self.eat()

        done = self.is_alive()

        state = []
        state += self.neighbors
        state += [self.hunger]

        return state, self.reward, done, {}

    def reset(self):
        self.__init__()

        self.scan()
        state = []
        state += self.neighbors
        state += [self.hunger]
        return state

    def next_day(self):
        self.world.make_food(1)
        if self.hunger == 0:
            self.health -= 25
        else:
            self.hunger -= 10

    def is_alive(self):
        return not self.health

    def eat(self):
        if self.food == 0:
            self.reward = -1.0
            return

        self.hunger += 50
        self.food -= 1
        self.reward = 1.0

    def move(self, direction):

        # time.sleep(1)
        if direction == "UP":
            if self.loc[0] == 0:
                return
            self.loc[0] -= 1
        if direction == "DOWN":
            if self.loc[0] == self.world.height - 1:
                return
            self.loc[0] += 1
        if direction == "LEFT":
            if self.loc[1] == 0:
                return
            self.loc[1] -= 1
        if direction == "RIGHT":
            if self.loc[1] == self.world.width - 1:
                return
            self.loc[1] += 1

        self.world.update_player(self)
        return

    def scan(self):
        def isNone(neighbor):
            if neighbor[0] < 0:
                return None
            if neighbor[0] > self.world.height - 1:
                return None
            if neighbor[1] < 0:
                return None
            if neighbor[1] > self.world.width - 1:
                return None
            return neighbor

        self.neighbors = [
            [self.loc[0]-1, self.loc[1]],
            [self.loc[0]+1, self.loc[1]],
            [self.loc[0], self.loc[1]-1],
            [self.loc[0], self.loc[1]+1]
        ]

        self.neighbors = [isNone(neighbor) for neighbor in self.neighbors]
        self.neighbors = [False if neighbor is None else
                    self.world.map[neighbor[0]][neighbor[1]] == "O" for neighbor in self.neighbors]





if __name__ == '__main__':
    p1 = Player()
    p1.next_day()