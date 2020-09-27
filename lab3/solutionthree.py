from misio.lost_wumpus.agents import AgentStub
from misio.lost_wumpus import Action
from misio.optilio.lost_wumpus import run_agent
from scipy import signal
import numpy as np


class MyAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)

        self.histogram = np.full(self.map.shape, 1 / (self.h * self.w))

        self.exit = np.argwhere(self.map == 2).flatten()

        self.kernel = np.array([[0, (1-self.p)/4, 0],
                                [(1-self.p)/4, self.p, (1-self.p)/4],
                                [0, (1-self.p)/4, 0]])

    def sense(self, sensory_input: bool):
        if sensory_input:
            self.histogram = np.vectorize(self.posterior_cave)(self.map, self.histogram)
        else:
            self.histogram = np.vectorize(self.posterior_no_cave)(self.map, self.histogram)

        self.histogram[self.exit[0], self.exit[1]] = 0
        self.normalize()

    def move(self):
        action = self.choose_action()
        self.histogram = signal.convolve2d(self.histogram, self.kernel, mode='same', boundary='wrap')
        self.roll(action)

        self.histogram[self.exit[0], self.exit[1]] = 0
        self.normalize()

        return action

    def get_histogram(self):
        return self.histogram

    def reset(self):
        self.exit = np.argwhere(self.map == 2).flatten()
        self.histogram = np.full((self.h, self.w), 1 / (self.h * self.w))

    def normalize(self):
        s = np.sum(self.histogram)
        self.histogram = self.histogram / s

    def roll(self, action):
        if action == Action.LEFT:
            self.histogram = np.roll(self.histogram, -1, axis=1)
        elif action == Action.RIGHT:
            self.histogram = np.roll(self.histogram, 1, axis=1)
        elif action == Action.UP:
            self.histogram = np.roll(self.histogram, -1, axis=0)
        elif action == Action.DOWN:
            self.histogram = np.roll(self.histogram, 1, axis=0)

    def posterior_cave(self, field, cp):
        return (self.pj if field == 1.0 else self.pn) * cp

    def posterior_no_cave(self, field, cp):
        return ((1 - self.pj) if field == 1.0 else (1 - self.pn)) * cp

    def choose_action(self):
        position = np.unravel_index(self.histogram.argmax(), self.histogram.shape)
        diff = self.exit - position
        return self.map_diff_to_action(diff, position)

    def map_diff_to_action(self, diff, position):
        if (np.random.rand(1) < 0.5 and diff[1] != 0) or diff[0] == 0:
            if diff[1] > 0:
                if (position[1] + self.w - self.exit[1]) < (self.exit[1] - position[1]):
                    return Action.LEFT
                else:
                    return Action.RIGHT
            elif diff[1] < 0:
                if self.w - position[1] + self.exit[1] < position[1] - self.exit[1]:
                    return Action.RIGHT
                else:
                    return Action.LEFT
        else:
            if diff[0] > 0:
                if position[0] + self.h - self.exit[0] < self.exit[0] - position[0]:
                    return Action.UP
                else:
                    return Action.DOWN
            elif diff[0] < 0:
                if self.h - position[0] + self.exit[0] < position[0] - self.exit[0]:
                    return Action.DOWN
                else:
                    return Action.UP

        return np.random.choice(Action)






