#!/usr/bin/python3

import numpy as np
import itertools
from itertools import chain
from misio.uncertain_wumpus.testing import load_world

PREC = '.2f'

# generate all pit possibilities given the observation
# (A) calc prob for each one - then sum up
# (B) calc prob for each combination with pit on queried cell - sum up 
# prob for a pit in queried cell is B/A

class Query:
    def __init__(self, world, p):
        self.p = p
        self.world = world
        self.fringe = []
        self.combinations = []
        self.probabilities = []
        self.sum_of_probabilities = 1  
     
    def find_fields_to_calculate(self):
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '-1.0': self.fringe.append((i,j))
        
        print("fringe: \n", self.fringe)

    def generate_all_combinations(self):
        for L in range(1, len(self.fringe)+1):
            for subset in itertools.combinations(self.fringe, L):
                self.combinations.append(subset)

        print("combinations: \n", self.combinations)

    def probability_for_combinations(self):
        max_num_of_pits = len(self.fringe)
        for each in self.combinations:
            num_of_pits = len(each)
            num_of_empties = max_num_of_pits - num_of_pits
            probability = round(p**num_of_pits * (1-p)**num_of_empties,4)
            self.probabilities.append(probability)
        self.sum_of_probabilities = sum(self.probabilities)

        print("probabilities: \n", self.probabilities)



def print_result(lines, output_file):
    result = "\n".join([" ".join([str(x) for x in line]) for line in lines])
    print(result, file=output_file, flush=True)


def prewumpus(world, p):
    '''return matrix with 0.20 when unknown (can't calculate), 
    0 when known (breeze B or visited O) and -1 for fields to calculate'''    
    x = len(world[1])
    y = len(world)
    p = format(p, PREC)

    ret = np.full((y, x), p)
    # add 0-padding
    world = np.pad(world, pad_width=1, mode='constant', constant_values=0)
    ret = np.pad(ret, pad_width=1, mode='constant', constant_values=0) 

    for i in range(1,y+1):
        for j in range(1,x+1): 
            if world[i][j] == 'B':
                ret[i][j] = format(1, PREC)
                if ret[i-1][j]==p: ret[i-1][j] = -1.0
                if ret[i+1][j]==p: ret[i+1][j] = -1.0
                if ret[i][j-1]==p: ret[i][j-1] = -1.0
                if ret[i][j+1]==p: ret[i][j+1] = -1.0

            elif world[i][j] == 'O':
                ret[i][j] = format(0, PREC)
    
    # remove padding
    return ret[1:-1, 1:-1]


def wumpus(world,p):
    query = Query(world, p)
    query.find_fields_to_calculate()
    query.generate_all_combinations()
    query.probability_for_combinations()

    print("board:")

    return world


if __name__ == "__main__":
    import sys

    # input_file = sys.stdin # uncomment
    input_file = f = open("test_cases/2020_short.in", "r") # del
    output_file = sys.stdout
    
    instances_num = int(input_file.readline())
    for _ in range(instances_num):
        world, p = load_world(input_file)
        # wumups
        world = prewumpus(world, p)
        lines = wumpus(world, p)
        # out
        print_result(lines, output_file)
        print("--------------") # del

