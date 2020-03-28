#!/usr/bin/python3


from misio.uncertain_wumpus.testing import load_world
import numpy as np
import itertools, time
from itertools import chain
from operator import itemgetter
import signal

PREC = '.2f'
TIMEOUT = 5
SWITCH = 5

# TODO:
# add file checker
# fix correctness

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


class Wumpus:
    def __init__(self, world, p):
        self.p = np.float128(p)
        self.world = world

        self.breezes = set()
        self.breezes_len = 0

        self.fringe = []
        self.fringe_len = 0

        self.fringe_brezze_sets = {}
        self.combinations_per_fringe = {}

        self.probabilities = []
        self.sum_of_probabilities = np.float128(1)  

        self.x = 0
        self.y = 0

        self.fringes_to_check = set()
        self.breezes_to_check = set()

        self.visited = []


    def find_fields_to_calculate(self):
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '-1.0': 
                    self.fringe.append((i,j))
                    self.fringes_to_check.add((i,j)) #new

                    breezes_copy = set()
                    if i+1 <= y: breezes_copy.add((i+1,j))
                    if i-1 >= 0: breezes_copy.add((i-1,j))
                    if j+1 <= x: breezes_copy.add((i,j+1))
                    if j-1 >= 0: breezes_copy.add((i,j-1))
                    self.fringe_brezze_sets[(i,j)] = breezes_copy

                if self.world[i][j] == '2.00': 
                    self.breezes.add((i,j))
                    self.breezes_to_check.add((i,j)) 

        self.fringe_len = len(self.fringe)
        self.breezes_len = len(self.breezes)
        self.combinations_per_fringe = dict([(key, []) for key in self.fringe])

        self.x = len(self.world[1])
        self.y = len(self.world)

        # print("fringe: \n", self.fringe)


    def find_front(self, field, local_breezes, local_fringes):
        i, j = field

        right = (i+1,j) if i+1 <= self.y    else None
        left =  (i-1,j) if i-1 >= 0         else None
        up =    (i,j+1) if j+1 <= self.x    else None
        down =  (i,j-1) if j-1 >= 0         else None
                

        for way in [right, left, up, down]:
            if way is not None:

                if way in self.breezes_to_check: 
                    local_breezes.add(way)
                    self.breezes_to_check.remove(way)
                    self.find_front(way, local_breezes, local_fringes)

                elif way in self.fringes_to_check:
                    local_fringes.add(way)
                    self.fringes_to_check.remove(way)
                    self.find_front(way, local_breezes, local_fringes)

        # print(field, " finished")
        return list(local_fringes)

    def check_indexes(self, f_indexes, local_group, local_breezes):
        ''' combination should cover all B (==2.0) fields in group '''

        comb_list = set()
        for each in f_indexes:
            comb_list |= self.fringe_brezze_sets[local_group[each]] # get covered-breezes set for each fringe in group

        if local_breezes.issubset(comb_list): return True
        else: return False


    def check_list(self, f_list, local_group, local_breezes):
        ''' combination should cover all B (==2.0) fields in group '''

        comb_list = set()
        for each in f_list:
            comb_list |= self.fringe_brezze_sets[each] # get covered-breezes set for each fringe in group

        if local_breezes.issubset(comb_list): return True
        else: return False


    def support(self, local_group, local_breezes, local_combinations = [], obligatory=[], added=0):
        added = 0
        fcopy = local_group.copy()
        breezes_len = len(local_breezes)

        for f in local_group:
            if f not in obligatory:
                fcopy.remove(f)

                fset = set(fcopy)
                fset_len = len(fset)


                if fset not in self.visited:               
                    if self.check_list(fcopy, local_group, local_breezes):
                        index = len(local_combinations)
                        for each in fset: 
                            self.combinations_per_fringe[each].append(index)

                        local_combinations.append(tuple(fset))
                        if fset_len*4 >= breezes_len and fset_len > 1: 
                            local_combinations = self.support(fcopy, local_breezes, local_combinations, obligatory, added)

                    else: 
                        if fset_len*4 >= breezes_len and fset_len > 1: 
                            obligatory.append(f)
                            added -= 1

                    self.visited.append(fset)

                fcopy.append(f)

            obligatory = obligatory[:added]

        return local_combinations

    def combination_for_full_front(self, local_group, local_combinations):
        index = len(local_combinations)
        for each in local_group: 
            self.combinations_per_fringe[each].append(index)

        local_combinations.append(tuple(local_group))
        
        return local_combinations


    def generate_all_combinations(self, local_group, local_breezes):
        ''' create list of combinations of all possible pits on board
        for each element in self.fringe create list of indexes to 
        particular combinations which contain this element '''

        group_len = len(local_group)
        breezes_len = len(local_breezes)
        local_combinations = []
        j = 0

        for i in range(1, group_len+1):
            if i*4 >= breezes_len:
                for subset in itertools.combinations(range(group_len), i):
                    if self.check_indexes(subset, local_group, local_breezes):
                        if i == 1:  comb_list = (local_group[subset[0]],)
                        else:       comb_list = itemgetter(*subset)(local_group)

                        local_combinations.append(comb_list)
                        # update list of combination for each fringe
                        [self.combinations_per_fringe[local_group[each]].append(j) for each in subset] 
                        j += 1

        return local_combinations


    def dododo(self):
        self.find_fields_to_calculate()
        signal.alarm(TIMEOUT)  
        try:
            for field in (self.fringe+list(self.breezes)):
                local_breezes = set()
                local_group = set()

                local_group_list = self.find_front(field, local_breezes, local_group)

                if len(local_group) >= SWITCH: 
                    # print(">>>>>>>>>>>>>> SUPPORT")
                    local_combinations = self.support(local_group_list, local_breezes, [], [], 0)
                    local_combinations = self.combination_for_full_front(local_group_list, local_combinations)
                else:
                    local_combinations = self.generate_all_combinations(local_group_list, local_breezes)

                local_probabilities = self.probabilities_for_group(local_group_list, local_combinations)

                self.calculate_probabilities(local_group_list, local_probabilities)

                # print("breezes: ", local_breezes)
                # print("local group: ", local_group)            
                # print("combinations: ", local_combinations)
                # print("probabilities: ", local_probabilities)

                if not self.fringes_to_check and not self.breezes_to_check: break

            self.change_breeze_fileds_to_zero()
        except TimeoutException:
            # print("----- EXCEPTION")
            self.change_breeze_fileds_to_zero()

    def probabilities_for_group(self, local_group, combinations):
        probabilities = []
        max_num_of_pits = len(local_group)

        for combination in combinations:
            num_of_pits = len(combination)
            num_of_empties = max_num_of_pits - num_of_pits
            probability = np.multiply(
                np.power(self.p, num_of_pits, dtype=np.float128), 
                np.power(1-self.p, num_of_empties, dtype=np.float128)
                )
            probabilities.append(probability)

        return probabilities

    # change to normalization method
    def calculate_probabilities(self, local_group, probabilities):
        sum_of_all = np.sum(probabilities)

        for fringe in local_group:
            if sum_of_all == 0: 
                q = np.float128(0)
            else:
                indexes = self.combinations_per_fringe[fringe]
                if len(indexes) == 1: sum_of_fringe = probabilities[indexes[0]]
                else: sum_of_fringe = sum(itemgetter(*indexes)(probabilities))                 
                q = np.divide(sum_of_fringe, sum_of_all, dtype=np.float128)

            q = np.around(q, decimals=2)
            x, y = fringe
            self.world[x][y] = format(q, PREC)


    def change_breeze_fileds_to_zero(self):
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '1.0':
                    self.world[i][j] = format(1, PREC)
                if self.world[i][j] == '2.00':
                    self.world[i][j] = format(0, PREC)
                elif self.world[i][j] == '-1.0':
                    self.world[i][j] = format(self.p, PREC)


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




if __name__ == "__main__":
    import sys

    

    input_file = sys.stdin # uncomment
    # input_file = f = open("test_cases/2019_02_medium.in", "r") # del
    # input_file = f = open("test_cases/2020_short.in", "r") # del
    output_file = sys.stdout

    instances_num = int(input_file.readline())
    for _ in range(instances_num):
        world, p = load_world(input_file)
        # wumups
        world = pre_wumpus(world, p)
        wumpus = Wumpus(world, p)

        # t1 = time.time()
        wumpus.dododo()
        # print(time.time() - t1)

        world = wumpus.get_world()
        # out
        print_result(world, output_file)

