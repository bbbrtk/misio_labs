import numpy as np
import sys
from math import factorial
from functools import lru_cache

TEST = False

# @lru_cache(maxsize=None)
def poisson(_lambda: float, n: int):
    return _lambda ** n * np.e ** (-_lambda) / factorial(n)


def get_poisson_probabilities(_lambda: int, n: int):
    density = 1.0
    poisson_results = []
    for i in range(n + 1):
        val = poisson(_lambda, i)
        density -= val
        poisson_results.append((val, density, val + density))
    return poisson_results

class State:
    def __init__(self, probability, state):
        self.probability = probability
        self.state = state
        self.gain = 0


class ShopsState:
    def __init__(self, s1_state: State, s2_state: State):
        self.s1_state = s1_state
        self.s2_state = s2_state

class StoreAgent(object):
    def __init__(self):
        self.gamma = 0.0
        self.m = 0
        self.g = 0
        self.c = 0
        self.f = 0
        # number of guests
        self.l1 = 0
        self.l2 = 0
        # number of picked mushrooms
        self.l3 = 0
        self.l4 = 0
        # arrays
        self.s1_clients = []
        self.s2_clients = []
        self.picked_mushrooms_s1 = []
        self.picked_mushrooms_s2 = []
        # other
        self.expected_gain = dict()
        self.next_states = dict()
        self.possible_actions = []

    def _get_poissons_probabilities(self):
        self.s1_clients = get_poisson_probabilities(self.l1, self.m)
        self.s2_clients = get_poisson_probabilities(self.l2, self.m)
        self.picked_mushrooms_s1 = get_poisson_probabilities(self.l3, self.m)
        self.picked_mushrooms_s2 = get_poisson_probabilities(self.l4, self.m)

    def _get_gain(self, number_of_mushrooms):
        return self.g * number_of_mushrooms

    def _get_moving_cost(self, number_of_mushrooms):
        if number_of_mushrooms < self.f:
            return self.c * number_of_mushrooms
        else:
            return self.c * self.f

    def _get_f_range(self):
        return [range(-self.f, self.f)]


    # -----------------------------------
    # get probabilities for all grow states
    def generate_possible_grow(self, state: int, grow_distribution):
        max_number_to_grow = self.m - state + 1
        # returns an array, if gorw_count == state then add state's probability and
        # the remaining probability
        return [State(grow_distribution[grow_count][2], grow_count)
                if state == self.m else State(grow_distribution[grow_count][0], grow_count)
                for grow_count in range(max_number_to_grow)]

    # get probabilities for all sales states
    def generate_possible_sales(self, state: int, shop_distribution):
        max_number_to_sale = state + 1
        # returns an array, if sell_count == state then add state's probability and
        # the remaining probability
        return [State(shop_distribution[sell_count][2], sell_count)
                if sell_count == state else State(shop_distribution[sell_count][0], sell_count)
                for sell_count in range(max_number_to_sale)]
    # -----------------------------------


    def run(self):
        # return self.simulation()
        self._get_poissons_probabilities()
        arr = np.zeros((self.m + 1, self.m + 1))

        return arr

def run_agent():
    n_worlds = int(input())
    for world_i in range(n_worlds):
        agent = StoreAgent()
        agent.m = int(input())
        agent.gamma = float(input())
        agent.l1, agent.l2, agent.l3, agent.l4 = [int(x) for x in input().split()]
        agent.g = int(input())
        agent.c = int(input())
        agent.f = int(input())
        agent.run()


if __name__ == '__main__':
    if TEST:
        pass
    else:
        run_agent()
