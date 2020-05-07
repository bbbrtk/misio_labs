import numpy as np
import sys
from math import factorial
from functools import lru_cache

TEST = True

# @lru_cache(maxsize=None)
def poisson(lab: float, n: int):
    return lab ** n * np.e ** (-lab) / factorial(n)


class StoreWumpus(object):
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


    def run(self):
        finalmatrix = np.zeros((self.m + 1, self.m + 1))

        runs = self.m * 1000 / 2
        
        store_1 = np.random.poisson(self.l1, runs)
        store_2 = np.random.poisson(self.l2, runs)
        shrooms_1 = np.random.poisson(self.l3, runs)
        shrooms_2 = np.random.poisson(self.l4, runs)

        state = self.m + 1

        for s1 in range(state):
            for s2 in range(state):

                results = np.zeros((state * 2))

                for move_shrooms in range(-self.f, self.f):
                    utility = 0
                    for i in range(runs):
                        if move_shrooms > 0 and s1 < move_shrooms or move_shrooms < 0 and s2 < abs(move_shrooms):
                            continue

                        calc = min(store_1[i], max(0, s1 - move_shrooms)) * self.g \
                                + min(store_2[i], max(0, s2 + move_shrooms)) * self.g \
                                - abs(move_shrooms) * self.c
                        utility += calc

                    results[move_shrooms + self.f] += utility

                best_value = np.argmax(results)
                finalmatrix[s1, s2] = best_value if best_value != 0 else self.f

        finalmatrix -= self.f

        print('\n'.join([' '.join([str(cell)[:-2] for cell in row]) for row in finalmatrix]))
        return finalmatrix

def test():
    agent = StoreWumpus()
    agent.m = 4
    agent.gamma = 0.95
    agent.l1 = 1
    agent.l2 = 2
    agent.l3 = 3
    agent.l4 = 4
    agent.g = 5
    agent.c = 2
    agent.f = 4

    agent._get_poissons_probabilities()
    
    for s1 in range(agent.m + 1):
        for s2 in range(agent.m + 1):
            sale_profit = s1*agent.g + s2*agent.g
            # p3% szans ze w sklepie s1 wyrosnie Y1 grzybow
            # p4% szans ze w sklepie s2 wyrosnie Y2 grzybow

            # p1% szans ze w sklepie s1 bedzie X1 gosci -> wtedy zarobek to min(X1, Y1) * g
            # p2% szans ze w sklepie s2 bedzie X2 gosci -> wtedy zarobek to min(X2, Y2) * g

            # mozna zwiekszyc zarobek przesuwajac Z <= f grzybow:
            # koszt to Z*c



def run_agent():
    n_worlds = int(input())
    for world_i in range(n_worlds):
        agent = StoreWumpus()
        agent.m = int(input())
        agent.gamma = float(input())
        agent.l1, agent.l2, agent.l3, agent.l4 = [int(x) for x in input().split()]
        agent.g = int(input())
        agent.c = int(input())
        agent.f = int(input())
        agent.run()


if __name__ == '__main__':
    if TEST:
        test()
    else:
        run_agent()
