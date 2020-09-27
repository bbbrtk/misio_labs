import numpy as np
import scipy

from misio.lost_wumpus._wumpus import Action, Field
from misio.lost_wumpus.agents import AgentStub
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cityblock
from misio.optilio.lost_wumpus import run_agent

class MyAgent(AgentStub):
    def __init__(self, map: np.ndarray, p: float, pj: float, pn: float):
        super().__init__(map, p, pj, pn)

        self.oposite_actions = {
            Action.DOWN.value: Action.UP,
            Action.UP.value: Action.DOWN,
            Action.RIGHT.value: Action.LEFT,
            Action.LEFT.value: Action.RIGHT
        }

        self.holes_num = np.sum(map == 1)
        self.empty_num = map.size - self.holes_num - 1
        self.exit_idxs = np.argwhere(self.map == 2)

        self.histogram = np.ones_like(map) / (map.size - 1)
        for exit_idx in self.exit_idxs:
            self.histogram[tuple(exit_idx)] = 0

        self.move_probability = np.zeros((3, 3), dtype=np.float32)  # kernel for convolution
        self.move_probability[1, 1] = p
        self.neighbour_shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.neighbour_move_prob_idxs = np.array([np.array(row) for row in self.neighbour_shifts]) + 1
        for neighbour_place in self.neighbour_move_prob_idxs:
            self.move_probability[tuple(neighbour_place)] = (1 - p) / 4

        self._calculate_state_utilities()

        self.actions_history = [Action.UP, Action.LEFT, Action.DOWN, Action.UP]

    def reset(self):
        super().reset()
        self.holes_num = np.sum(self.map == 1)
        self.empty_num = self.map.size - self.holes_num - 1
        self.exit_idxs = np.argwhere(self.map == 2)

        self.histogram = np.ones_like(self.map) / (self.map.size - 1)
        for exit_idx in self.exit_idxs:
            self.histogram[tuple(exit_idx)] = 0

        self.move_probability = np.zeros((3, 3), dtype=np.float32)  # kernel for convolution
        self.move_probability[1, 1] = self.p
        self.neighbour_shifts = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.neighbour_move_prob_idxs = np.array([np.array(row) for row in self.neighbour_shifts]) + 1
        for neighbour_place in self.neighbour_move_prob_idxs:
            self.move_probability[tuple(neighbour_place)] = (1 - self.p) / 4

        self._calculate_state_utilities()

        # self.actions_history = [Action.UP, Action.LEFT, Action.DOWN, Action.UP]

    def sense(self, sensory_input: bool):
        # sensory_input
        # 1 - jama
        # 0 - brak jamy
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if (i, j) not in self.exit_idxs:
                    if 1 == sensory_input == self.map[i, j]:
                        self.histogram[i, j] *= self.pj
                    elif 1 == sensory_input != self.map[i, j]:
                        self.histogram[i, j] *= self.pn
                    elif 0 == sensory_input == self.map[i, j]:
                        self.histogram[i, j] *= 1 - self.pn
                    elif 0 == sensory_input != self.map[i, j]:
                        self.histogram[i, j] *= 1 - self.pj
        self.histogram /= np.sum(self.histogram)

    def _roll_histogram(self, action: Action):
        if action.value == Action.UP.value:
            return np.roll(self.histogram, -1, axis=0)
        if action.value == Action.DOWN.value:
            return np.roll(self.histogram, 1, axis=0)
        if action.value == Action.LEFT.value:
            return np.roll(self.histogram, -1, axis=1)
        return np.roll(self.histogram, 1, axis=1)

    def _chose_action(self):
        actions = [Action.DOWN, Action.UP, Action.RIGHT, Action.LEFT]
        action_values = np.zeros_like(actions)
        shifted_hists = []

        for action_idx, action in enumerate(actions):
            rolled_histogram = self._roll_histogram(action)
            shifted_hist = convolve(rolled_histogram, self.move_probability, mode='wrap')
            utils = np.multiply(shifted_hist, self.state_utilities)
            action_values[action_idx] = np.sum(utils)
            shifted_hists.append(shifted_hist)

        # for action_idx in np.argsort(action_values)[::-1]:
        #     if self.oposite_actions[self.actions_history[-1].value].value != actions[action_idx].value\
        #             and self.actions_history[-1].value != actions[action_idx].value:
        #         self.actions_history.pop(0)
        #         self.actions_history.append(actions[action_idx])
        #         return actions[action_idx], shifted_hists[action_idx]

        best_idx = int(np.argmax(action_values))

        # self.state_utilities -= 0.001 * self.histogram

        return actions[best_idx], shifted_hists[best_idx]

    def move(self):
        # action = np.random.choice(Action)
        action, self.histogram = self._chose_action()

        for exit_idx in self.exit_idxs:
            self.histogram[tuple(exit_idx)] = 0
        return action

    def get_histogram(self):
        return self.histogram

    def _cyclic_norm(self, idx, norm=cityblock):
        idx = np.array(idx)
        dists = []
        for exit_idx in self.exit_idxs:
            dists += [
                norm(exit_idx, idx),
                norm(exit_idx, idx - (len(self.map[0]), 0)),
                norm(exit_idx, idx - (0, len(self.map))),
                norm(exit_idx, idx - (len(self.map[0]), len(self.map)))]
        return np.sqrt(min(dists))

    def _calculate_state_utilities(self, num_iter=10):
        self.state_utilities = np.zeros_like(self.map, dtype=np.float32)

        self.state_utilities[self.map == 2] = 1.

        self.state_utilities = \
            np.array([np.array([-self._cyclic_norm((i, j)) if self.map[i, j] == 1 else -self._cyclic_norm((i, j))
                                for j, value in enumerate(row)])
                      for i, row in enumerate(self.state_utilities)])

        pass

if __name__ == "__main__":
    run_agent(MyAgent)