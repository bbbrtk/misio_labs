import numpy as np

from misio.lost_wumpus._wumpus import Action, Field
from misio.lost_wumpus.agents import AgentStub

from scipy.signal import convolve2d
from scipy.spatial.distance import cityblock
from misio.optilio.lost_wumpus import run_agent

class FinalAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(FinalAgent, self).__init__(*args, **kwargs)

        self.oposite_actions = {
            Action.DOWN.value: Action.UP,
            Action.UP.value: Action.DOWN,
            Action.RIGHT.value: Action.LEFT,
            Action.LEFT.value: Action.RIGHT
        }

        self.stat_helper = 4
        self.histogram = np.full(self.map.shape, 1 / (self.h * self.w))
        self.hist_helper = np.ones_like(self.map) / (self.map.size - 1)
        # self.histogram = np.ones_like(map) / (map.size - 1)
        self.holes_num = np.sum(self.map == 1)
        self.empty_num = self.map.size - self.holes_num - 1
        self.exit_idxs = np.argwhere(self.map == 2)
        self.exit = np.argwhere(self.map == 2).flatten()
        self.kernel = np.array([[0, (1-self.p)/self.stat_helper, 0],[(1-self.p)/self.stat_helper, self.p, (1-self.p)/self.stat_helper],[0, (1-self.p)/self.stat_helper, 0]])
        self.random_vars = 0.65

    def sense(self, sensory_input: bool):
        # sensory_input
        # 1 - jama
        # 0 - brak jamy
        if sensory_input:
            self.histogram = np.vectorize(self.update)(self.map, self.histogram)
        else:
            self.histogram = np.vectorize(self.no_update)(self.map, self.histogram)

        self.histogram[self.exit[0], self.exit[1]] = 0
        self.histogram /= np.sum(self.histogram)

    def move(self):
        action = self.choose_action()
        self.histogram = convolve2d(self.histogram, self.kernel, mode='same', boundary='wrap')
        self.roll(action)
        # self._roll_histogram(action)

        self.histogram[self.exit[0], self.exit[1]] = 0
        self.histogram /= np.sum(self.histogram)

        return action

    def get_histogram(self):
        return self.histogram

    def get_hist_helper(self):
        return self.hist_helper

    def reset(self):
        self.holes_num = np.sum(self.map == 1)
        self.empty_num = self.map.size - self.holes_num - 1
        self.exit = np.argwhere(self.map == 2).flatten()
        self.hist_helper = np.ones_like(self.map) / (self.map.size - 1)
        self.histogram = np.full((self.h, self.w), 1 / (self.h * self.w))

    def roll(self, action):
        roll_left = np.roll(self.histogram, -1, axis=1)
        roll_right = np.roll(self.histogram, 1, axis=1)
        roll_up = np.roll(self.histogram, -1, axis=0)
        roll_down = np.roll(self.histogram, 1, axis=0)

        if action == Action.LEFT:
            self.histogram = roll_left
        elif action == Action.RIGHT:
            self.histogram = roll_right
        elif action == Action.UP:
            self.histogram = roll_up
        elif action == Action.DOWN:
            self.histogram = roll_down

    def _roll_histogram(self, action: Action):
        if action.value == Action.UP.value:
            return np.roll(self.histogram, -1, axis=0)
        elif action.value == Action.DOWN.value:
            return np.roll(self.histogram, 1, axis=0)
        elif action.value == Action.LEFT.value:
            return np.roll(self.histogram, -1, axis=1)
        return np.roll(self.histogram, 1, axis=1)

    def choose_action(self):
        pos = np.unravel_index(self.histogram.argmax(), self.histogram.shape)
        ex_pos = self.exit - pos
        self.hist_helper = np.ones_like(self.map) / (self.map.size - 1)
        act_mapping = self.action_mapping_new(ex_pos, pos)
        return act_mapping

    def _cyclic_norm(self, idx, norm=cityblock):
        idx = np.array(idx)
        dists = []
        for indexes_of_exit in self.exit_idxs:
            dists += [
                norm(indexes_of_exit, idx),
                norm(indexes_of_exit, idx - (len(self.map[0]), 0)),
                norm(indexes_of_exit, idx - (0, len(self.map))),
                norm(indexes_of_exit, idx - (len(self.map[0]), len(self.map)))]
        return np.sqrt(min(dists))

    def action_mapping_new(self, action_variance, index_current_pos):
        if (np.random.rand(1) < self.random_vars and action_variance[1] != 0) or action_variance[0] == 0:
            if action_variance[1] > 0:
                if (index_current_pos[1] + self.w - self.exit[1]) < (self.exit[1] - index_current_pos[1]):
                    return Action.LEFT
                else:
                    return Action.RIGHT
            elif action_variance[1] <= 0:
                if self.w - index_current_pos[1] + self.exit[1] < index_current_pos[1] - self.exit[1]:
                    return Action.RIGHT
                else:
                    return Action.LEFT
        else:
            if action_variance[0] > 0:
                if index_current_pos[0] + self.h - self.exit[0] < self.exit[0] - index_current_pos[0]:
                    return Action.UP
                else:
                    return Action.DOWN
            elif action_variance[0] <= 0:
                if self.h - index_current_pos[0] + self.exit[0] < index_current_pos[0] - self.exit[0]:
                    return Action.DOWN
                else:
                    return Action.UP

        return np.random.choice(Action)

    def update(self, field, cp):
        return (self.pj if field == 1.0 else self.pn) * cp

    def no_update(self, field, cp):
        return ((1 - self.pj) if field == 1.0 else (1 - self.pn)) * cp




if __name__ == "__main__":
    run_agent(FinalAgent)