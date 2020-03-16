#!/usr/bin/python3


from misio.uncertain_wumpus.testing import load_world
import numpy as np
import itertools
from itertools import chain

PREC = '.2f'

# TODO
'''
1. speed up generation of combination (generate only few combs) - like tree
2. measure duration of functions
3. delete some loops
'''

class Wumpus:
    def __init__(self, world, p):
        self.p = np.float128(p)
        self.world = world
        self.breezes = []
        self.visited = []
        self.fringe = []
        self.combinations = []
        self.probabilities = []
        self.dict_of_fringe_probs = {}
        self.sum_of_probabilities = np.float128(1)  
     

    def find_fields_to_calculate(self):
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '-1.0': self.fringe.append((i,j))
                if self.world[i][j] == '2.00': self.breezes.append((i,j))        
        # print("fringe: \n", self.fringe)


    def check_if_combination_is_sufficient(self, combination):
        ''' combination should cover all B (==2.0) fields on board '''
        breezes_copy = self.breezes.copy()
        # print(breezes_copy)
        for each in combination:
            x = each[0]
            y = each[1]
            if (x+1,y) in breezes_copy: breezes_copy.remove((x+1,y))
            if (x-1,y) in breezes_copy: breezes_copy.remove((x-1,y))
            if (x,y+1) in breezes_copy: breezes_copy.remove((x,y+1))
            if (x,y-1) in breezes_copy: breezes_copy.remove((x,y-1))

            if len(breezes_copy) == 0: 
                return True

        return False


    def generate_all_combinations(self):
        ''' create list of combinations of all possible pits on board
        for each element in self.fringe create list of indexes to 
        particular combinations which contain this element '''
        for each in self.fringe:
            self.dict_of_fringe_probs[each] = []

        j = 0
        for i in range(1, len(self.fringe)+1):
            for subset in itertools.combinations(self.fringe, i):
                if self.check_if_combination_is_sufficient(subset):
                    self.combinations.append(subset)
                    # print("sufficient combination: ", subset)
                    for each in subset:
                        self.dict_of_fringe_probs[each].append(j)
                    j += 1

        # print("combinations len: ", len(self.combinations))
        # print("combinations: \n", self.combinations)
        # print("dict_of_fringe_probs: \n", self.dict_of_fringe_probs)


    def probability_for_combinations(self):
        max_num_of_pits = np.float128(len(self.fringe))

        for combination in self.combinations:
            num_of_pits = np.float128(len(combination))
            num_of_empties = max_num_of_pits - num_of_pits

            probability = np.multiply(np.power(p, num_of_pits, dtype=np.float128), np.power(1-p, num_of_empties, dtype=np.float128))
            # print(probability, type(probability))
            self.probabilities.append(probability)

        self.sum_of_probabilities = np.sum(self.probabilities)
        # print("probabilities: \n", self.probabilities)


    def sum_up_probabilities(self, query):
        indexes = self.dict_of_fringe_probs[query]
        p_sum = np.float128(0)
        for index in indexes:
            p_sum += self.probabilities[index]
        return p_sum


    def calculate_probabilities(self):
        sum_of_all = np.sum(self.probabilities)
        # print("sum_of_all: ", sum_of_all)
        for each in self.fringe:
            sum_of_query = self.sum_up_probabilities(each)
            q = np.divide(sum_of_query, sum_of_all, dtype=np.float128)
            # print("field: ", each, " \t sum_of_query: ", round(sum_of_query, 4), "\t probability: ", q)
            q = np.around(q, decimals=2)
            self.world[each[0]][each[1]] = format(q, PREC)


    def change_breeze_fileds_to_zero(self):
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '1.0':
                    self.world[i][j] = format(1, PREC)
                if self.world[i][j] == '2.00':
                    self.world[i][j] = format(0, PREC)
                elif self.world[i][j] == '-1.O':
                    print(" !!! ERROR: -1 FOUND !!! ")

    def get_world(self):
        return self.world



def print_result(lines, output_file):
    result = "\n".join([" ".join([str(x) for x in line]) for line in lines])
    print(result, file=output_file, flush=True)


def pre_wumpus(world, p):
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
                ret[i][j] = format(2, PREC)
                if ret[i-1][j]==p: ret[i-1][j] = -1.0
                if ret[i+1][j]==p: ret[i+1][j] = -1.0
                if ret[i][j-1]==p: ret[i][j-1] = -1.0
                if ret[i][j+1]==p: ret[i][j+1] = -1.0

            elif world[i][j] == 'O':
                ret[i][j] = format(0, PREC)
                if ret[i-1][j] != '2.00': ret[i-1][j] = format(0, PREC)
                if ret[i+1][j] != '2.00': ret[i+1][j] = format(0, PREC)
                if ret[i][j-1] != '2.00': ret[i][j-1] = format(0, PREC)
                if ret[i][j+1] != '2.00': ret[i][j+1] = format(0, PREC)
    
    # remove padding
    # print(ret[1:-1, 1:-1])
    return ret[1:-1, 1:-1]


def wumpus_wumpus(world,p):
    wumpus = Wumpus(world, p)

    wumpus.find_fields_to_calculate()
    wumpus.generate_all_combinations()
    wumpus.probability_for_combinations()
    wumpus.calculate_probabilities()
    wumpus.change_breeze_fileds_to_zero()

    world = wumpus.get_world()
    # print("board:")

    return world


if __name__ == "__main__":
    import sys

    input_file = sys.stdin # uncomment
    # input_file = f = open("test_cases/2020_short.in", "r") # del
    output_file = sys.stdout
    
    instances_num = int(input_file.readline())
    for _ in range(instances_num):
        world, p = load_world(input_file)
        # wumups
        world = pre_wumpus(world, p)
        lines = wumpus_wumpus(world, p)
        # out
        print_result(lines, output_file)


