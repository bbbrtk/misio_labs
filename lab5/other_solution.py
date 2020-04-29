import numpy as np
import sys
from math import factorial
from functools import lru_cache

OPTILIO_MODE = False

DISTRIBUTION_VAL = 0
DISTRIBUTION_REMAINING = 0
DISTRIBUTION_VAL_REMAINING = 0


def print_numpy_array(array: np.ndarray):
    np.savetxt(sys.stdout.buffer, array, fmt='%d')


@lru_cache(maxsize=None)
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


class Action:
    def __init__(self, s1, s2, state, i):
        self.s1 = s1
        self.s2 = s2
        self.to_move = i
        self.gains = []
        self.state = state
    # def calc_future_gain(self, i, gamma, next_states):
    #     key = (self.s1, self.s2)
    #     if key in next_states:
    #         for next_states_list in next_states[key]:
    #             for next_state in next_states_list:

        return 0

    def calc_utility(self, i, gamma, g, c, next_states):
        return self.state.gain * g - abs(self.to_move) * c # + self.calc_future_gain(i, gamma, next_states) if i < 40 else 0

class StoreAgent(object):
    def __init__(self, m: int):
        self.m = m
        self.gamma = 0.0
        # number of guests
        self.l1 = 0
        self.l2 = 0
        # number of picked mushrooms
        self.l3 = 0
        self.l4 = 0
        self.g = 0
        self.c = 0
        self.f = 0
        self.shop_1_poisson = []
        self.shop_2_poisson = []
        self.picked_mushrooms_s1 = []
        self.picked_mushrooms_s2 = []
        self.expected_gain = dict()
        self.next_states = dict()
        self.possible_actions = []

    def get_poissons_probabilities(self):
        self.shop_1_poisson = get_poisson_probabilities(self.l1, self.m)
        self.shop_2_poisson = get_poisson_probabilities(self.l2, self.m)
        self.picked_mushrooms_s1 = get_poisson_probabilities(self.l3, self.m)
        self.picked_mushrooms_s2 = get_poisson_probabilities(self.l4, self.m)

    def get_reward(self, number_of_mushrooms):
        return self.c * number_of_mushrooms

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

    def generate_possible_states(self, state, shop_distribution, grow_distribution):
        '''
        Shop state = probabilities of all states from 0 to state (what is the probability of selling 0, 1,...n
        For each shop states we calculate grow state == how many mushrooms may grow taking into the consideration
        the current state and the number of sold mushrooms.
        :param state:
        :param shop_distribution:
        :param grow_distribution:
        :return: Array with states.
        '''
        actions = []
        for s1_shop_state in self.generate_possible_sales(state, shop_distribution):
            for s1_grow_state in self.generate_possible_grow(s1_shop_state.state, grow_distribution):
                gain = s1_shop_state.probability * s1_grow_state.probability * s1_shop_state.state
                actions.append((s1_shop_state.probability * s1_grow_state.probability, gain, s1_shop_state.state))
        return actions

    def generate_shops_state(self, s1_state: int, s2_state: int):
        s1_states = self.generate_possible_states(s1_state, self.shop_1_poisson, self.picked_mushrooms_s1)
        s2_states = self.generate_possible_states(s2_state, self.shop_2_poisson, self.picked_mushrooms_s2)
        for s1_state in s1_states:
            for s2_state in s2_states:
                # s1_state[0] = probability
                # s1_state[1] = gain
                # s1_state[2] = state
                key = (s1_state[2], s2_state[2])
                gain = s1_state[1] * s2_state[0] + s1_state[0] * s2_state[1]
                if key not in self.expected_gain:
                    self.expected_gain[key] = gain
                else:
                    self.expected_gain[key] += gain
                if key not in self.next_states:
                    self.next_states[key] = []
                self.next_states[key].append((s1_state[2], s2_state[2], gain, s1_state[0]*s2_state[0]))
        return ShopsState(s1_states, s2_states)

    def get_first_state(self, s1, s2, income):
        # to s1 we can move as many mushrooms as is in the s2, or as many as
        # it may be packed in s1 or no more than f
        to_move_to_s1 = min(s2, self.f, max(self.m - s1, 0))
        to_move_to_s2 = min(s1, self.f, max(self.m - s2, 0))

        for i in range(to_move_to_s1, -to_move_to_s2 - 1, 1):
            _s1 = min(s1 - i, self.m)
            _s2 = min(s2 + i, self.m)
            key = ()
            if key not in self.possible_actions:
                self.possible_actions[key] = []
            self.possible_actions.append(Action(_s1, _s2, income, i))

    def calc_result(self, s1, s2, income):
        self.possible_actions

        return 0

    def simulation(self):
        simulation_count = 1000
        l1 = np.random.poisson(self.l1, simulation_count)
        l2 = np.random.poisson(self.l2, simulation_count)
        u = dict()

        arr = np.zeros((self.m + 1, self.m + 1))

        for s1 in range(self.m + 1):
            for s2 in range(self.m + 1):
                results = np.zeros((2 * self.m + 1))
                for a in range(-self.f, self.f):
                    u = 0
                    for i in range(simulation_count):
                        if a > 0 and s1 < a or a < 0 and s2 < abs(a):
                            continue
                        u -= abs(a) * self.c
                        u += min(l1[i], max(0, s1 - a)) * self.g
                        u += min(l2[i], max(0, s2 + a)) * self.g
                    results[a + self.f] += u

                best_value = np.argmax(results)
                arr[s1, s2] = best_value if best_value != 0 else self.f

        arr -= self.f
        print_numpy_array(arr)
        return arr

    def run(self):
        # return self.simulation()
        self.get_poissons_probabilities()
        arr = np.zeros((self.m + 1, self.m + 1))

        night_state = np.zeros_like(arr)
        day_state = np.zeros_like(arr)

        income = []
        for s1 in range(self.m + 1):
            row = []
            for s2 in range(self.m + 1):
                row.append(self.generate_shops_state(s1, s2))
            income.append(row)
        for s1 in range(self.m+1):
            for s2 in range(self.m+1):
                self.get_first_state(s1, s2, income)
        for s1 in range(self.m+1):
            for s2 in range(self.m+1):
                arr[s1, s2] = self.calc_result(s1, s2, income)

        print_numpy_array(arr)
        return arr

def run_agent():
    n_worlds = int(input())
    for world_i in range(n_worlds):
        m = int(input())
        agent = StoreAgent(m)
        agent.gamma = float(input())
        agent.l1, agent.l2, agent.l3, agent.l4 = [int(x) for x in input().split()]
        agent.g = int(input())
        agent.c = int(input())
        agent.f = int(input())
        agent.run()


if __name__ == '__main__':
    if OPTILIO_MODE:
        run_agent()
    else:
        from store_agent_test import test_store_agent

        test_store_agent(print_invalid_coords=False, short_test=False)