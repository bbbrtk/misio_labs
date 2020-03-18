#!/usr/bin/python3


from misio.uncertain_wumpus.testing import load_world
import numpy as np
import itertools, time
from itertools import chain
from operator import itemgetter
import signal

PREC = '.2f'
TIMEOUT = 11

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
        self.visited = []
        self.fringe = [] # front
        self.fringe_len = 0
        self.fringe_brezze_sets = []
        self.combinations = []
        self.probabilities = []
        self.dict_of_fringe_probs = []
        self.sum_of_probabilities = np.float128(1)  
        # self.time1 = time.time()
        self.comb = []
        self.visited = []
        self.fringe_incombs = []
        self.probs = []
        self.sum_probs = np.float128(1)  

    def find_fields_to_calculate(self):
        # self.time1 = time.time()
        x = len(self.world[1])
        y = len(self.world)
        for i in range(y):
            for j in range(x): 
                if self.world[i][j] == '-1.0': 
                    self.fringe.append((i,j))
                    breezes_copy = set()
                    if i+1 <= y: breezes_copy.add((i+1,j))
                    if i-1 >= 0: breezes_copy.add((i-1,j))
                    if j+1 <= x: breezes_copy.add((i,j+1))
                    if j-1 >= 0: breezes_copy.add((i,j-1))
                    self.fringe_brezze_sets.append(breezes_copy)

                if self.world[i][j] == '2.00': self.breezes.add((i,j)) 

        self.fringe_len = len(self.fringe)
        self.breezes_len = len(self.breezes)
        self.dict_of_fringe_probs = [ [] for i in range(self.fringe_len) ]

        self.fringe_incombs = [ [] for i in range(self.fringe_len) ]

        # print("fringe: \n", self.fringe)

    
    def check_if_combination_is_sufficient(self, subset_indexes, i):
        ''' combination should cover all B (==2.0) fields on board '''
        breezes_copy = set()
        res_list = itemgetter(*subset_indexes)(self.fringe_brezze_sets)

        if i == 1:
            if self.breezes.issubset(res_list): return True
            else: return False

        else:
            for each in res_list:
                breezes_copy |= each
        
        if self.breezes.issubset(breezes_copy): return True
        else: return False


    def generate_all_combinations(self):
        ''' create list of combinations of all possible pits on board
        for each element in self.fringe create list of indexes to 
        particular combinations which contain this element '''
        t1 = time.time()
        j = 0
        # k = 0
        for i in range(1, self.fringe_len+1):
            # if time.time()-self.time1 > TIMEOUT: break
            if i*4 > self.breezes_len:
                for subset in itertools.combinations(range(self.fringe_len), i):
                    # k += 1
                    # if time.time()-self.time1 > TIMEOUT: break
                    if self.check_if_combination_is_sufficient(subset, i):
                        
                        if i == 1: res_list = (self.fringe[subset[0]],)
                        else: res_list = itemgetter(*subset)(self.fringe)

                        self.combinations.append(res_list)
                        # update list of probabilities per combination
                        [self.dict_of_fringe_probs[each].append(j) for each in subset] 
                        j += 1

        # print("generate_all_comb: ", j, "/", k, " - ", self.fringe_len, "\t", time.time()-t1)
        # print("generate_all_comb: ", j, " - ", self.fringe_len, "\t", time.time()-t1)
        # print("combinations len: ", len(self.combinations))
        # print("combinations: \n", self.combinations)
        # print("dict_of_fringe_probs: \n", self.dict_of_fringe_probs)


    def check(self, f_indexes):
        ''' combination should cover all B (==2.0) fields on board '''
        # breezes_copy = set()
        res_list = set()
        # res_list = itemgetter(*f_indexes)(self.fringe_brezze_sets)
        for each in f_indexes:
            res_list |= self.fringe_brezze_sets[each]


        # if len(res_list) == 1:
        #     if self.breezes.issubset(res_list): return True
        #     else: return False

        # else:
        #     for each in res_list:
        #         breezes_copy |= each
        
        if self.breezes.issubset(res_list): return True
        else: return False

    def support(self, flist, obligatory=[], added=0):
        ok = True
        added = 0
        fcopy = flist.copy()
        # print("flist: ", flist)

        for f in flist:
            # print(f, "_f : ", self.fringe[f], " obl: ", obligatory, " vis: ", self.visited)
            

            if f not in obligatory:
                fcopy.remove(f)
                
                # print("is ",  set(fcopy), " visited? ", set(fcopy) in self.visited)
                if set(fcopy) not in self.visited:
                                
                    
                    if self.check(fcopy):
                        # print("good flist: ", fcopy)
                        for each in fcopy: self.fringe_incombs[each].append(len(self.comb))
                        self.comb.append(set(fcopy))

                        # if len(fcopy)*4 > self.breezes_len and len(fcopy) > 1:  
                        if len(fcopy) > 1: 
                            # print("\n NEXT LEVEL vvv")
                            self.support(fcopy, obligatory, added)
                            # print("LEFT LEVEL")
                    else: 
                        if len(fcopy) > 1: 
                            obligatory.append(f)
                            # print("added oblig: ", f)
                            added -= 1
                

                    self.visited.append(set(fcopy))
                    # print("visited: ", set(fcopy) )
                fcopy.append(f)

            obligatory = obligatory[:added]
        # print("--- fin support --- \n")



    def probability_for_combinations(self):
        max_num_of_pits = np.float128(self.fringe_len)

        for combination in self.comb:
            num_of_pits = np.float128(len(combination))
            num_of_empties = max_num_of_pits - num_of_pits

            probability = np.multiply(np.power(p, num_of_pits, dtype=np.float128), np.power(1-p, num_of_empties, dtype=np.float128))
            # print(probability, type(probability))
            self.probabilities.append(probability)

        self.sum_of_probabilities = np.sum(self.probabilities)
        # print("probabilities: \n", self.probabilities)


    def sum_up_probabilities(self, query):
        indexes = self.fringe_incombs[query]
        p_sum = np.float128(0)
        for index in indexes:
            p_sum += self.probabilities[index]
        return p_sum


    def calculate_probabilities(self):
        sum_of_all = np.sum(self.probabilities)
        # print("sum_of_all: ", sum_of_all)
        for i in range(self.fringe_len):
            sum_of_query = self.sum_up_probabilities(i)
            if sum_of_all == 0:
                q = np.float128(0)
            else:
                q = np.divide(sum_of_query, sum_of_all, dtype=np.float128)
            # print("field: ", i, " \t sum_of_query: ", round(sum_of_query, 4), "\t probability: ", q)
            q = np.around(q, decimals=2)
            x = self.fringe[i][0]
            y = self.fringe[i][1]
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


    def do_wumpus(self):
        self.find_fields_to_calculate()
        
        signal.alarm(TIMEOUT)    
        # This try/except loop ensures that 
        #   you'll catch TimeoutException when it's sent.
        try:
            self.generate_all_combinations()
        except TimeoutException:
            self.change_breeze_fileds_to_zero()
        else:
            # Reset the alarm
            signal.alarm(0)
            self.probability_for_combinations()
            self.calculate_probabilities()
            self.change_breeze_fileds_to_zero()

    def do_wumpus_2(self):
        self.find_fields_to_calculate()

        self.comb.append(set(self.fringe))
        for each in self.fringe_incombs: each.append(0)

        self.support(list(range(self.fringe_len)))

        # print("self.comb \t", self.comb)

        self.probability_for_combinations()
        self.calculate_probabilities()
        self.change_breeze_fileds_to_zero()





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
    wumpus.do_wumpus_2()
    return wumpus.get_world()


if __name__ == "__main__":
    import sys

    # input_file = sys.stdin # uncomment
    input_file = f = open("test_cases/2020_short.in", "r") # del
    output_file = sys.stdout
    # t1 = time.time()
    instances_num = int(input_file.readline())
    for _ in range(instances_num):
        world, p = load_world(input_file)
        # wumups
        world = pre_wumpus(world, p)
        lines = wumpus_wumpus(world, p)
        # out
        print_result(lines, output_file)

    # print("all time: ",     time.time()-t1)
