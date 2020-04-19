#!/usr/bin/python3

from misio.lost_wumpus._wumpus import *
from misio.lost_wumpus.agents import *
from misio.lost_wumpus.util import load_input_file, load_world_from_stdin
from misio.optilio.lost_wumpus import run_agent
import numpy as np
import itertools, time  
from operator import itemgetter

import random
import sys

PREC = '.8f'
OUTPUT_FILE = sys.stdout
MY_PATH_MEMORY = 1
PERCENTAGE_OF_MOST_PROBABLE_POINTS = 1
PRINT_ALL = False


class OptWmp(object):
    def __init__(self, map: np.ndarray, p: float, pj: float, pn: float, exit_loc: tuple = None, max_moves=None):
        assert isinstance(map, np.ndarray)
        self.map = map
        self.board = map.astype('float64')
        self.h, self.w = map.shape
        self.p = p
        self.p_rest = (1-p)/4
        self.pj = pj
        self.pn = pn
        self.exit_border_h = ('LEFT', 0)
        self.exit_border_w = ('UP', 0)
        if exit_loc is None:
            exit_loc = [int(x) for x in np.where(map == Field.EXIT)]
            ey, ex = exit_loc
            left_ex = ex # abs(0 - ex)
            right_ex = abs(self.w - ex)
            if right_ex < left_ex:
                # exit is closer to right border
                self.exit_border_w = ('RIGHT', right_ex)
            else:
                self.exit_border_w = ('LEFT', left_ex)

            up_ey = ey # abs(0 - ey)
            down_ey = abs(self.h - ey)
            if down_ey < up_ey:
                # exit is closer to bottom border
                self.exit_border_h = ('DOWN', down_ey)
            else:
                self.exit_border_h = ('UP', up_ey)

        self.exit_loc = list(exit_loc)
        self.position = list(exit_loc)
        self.moves = np.inf
        self.finished = True
        self.sensory_output = None
        if max_moves is None:
            max_moves = np.inf
        self.max_moves = max_moves

        self.signal = self.position
        self.random_moves = []
        self.path = {Action.LEFT : 0, Action.DOWN : 0, Action.RIGHT : 0,  Action.UP : 0}
        self.last_move = Action.RIGHT
        self.my_path = []
        self.first = True

    def __str__(self):
        return super().__str__()

    def reset(self):
        self.moves = 0
        self.finished = False

    def sense(self, sensory_input: bool):
        self.signal = sensory_input

    def first_analyze(self):
        all_fields = self.h * self.w - 1
        field_prob = 1/all_fields
        
        for i in range(self.h):
            for j in range(self.w):
                self.board[i,j] = float(field_prob)

                if self.map[i,j] == Field.CAVE:     self.board[i,j] *= self.pj
                elif self.map[i,j] == Field.EMPTY:  self.board[i,j] *= self.pn
                elif self.map[i,j] == Field.EXIT:   self.board[i,j] = 0


    def cave_analyze(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.map[i,j] == Field.CAVE:     self.board[i,j] *= self.pj
                elif self.map[i,j] == Field.EMPTY:  self.board[i,j] *= self.pn
                elif self.map[i,j] == Field.EXIT:   self.board[i,j] = 0


    def empty_analyze(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.map[i,j] == Field.CAVE:     self.board[i,j] = 0
                # elif self.map[i,j] == Field.EMPTY:  self.board[i,j] *= self.pn
                elif self.map[i,j] == Field.EXIT:   self.board[i,j] = 0

    # check if this move is not in the opposite direction
    def block_counter_move(self, move):
        if     (move == Action.LEFT and self.last_move == Action.RIGHT) \
            or (move == Action.RIGHT and self.last_move == Action.LEFT) \
            or (move == Action.UP and self.last_move == Action.DOWN) \
            or (move == Action.DOWN and self.last_move == Action.UP):
            return True

        else: 
            return False

    # find manhatan distance for all most probable fields
    # sum up and choose the most often occured direction
    def select_move_direction(self):
        # if PRINT_ALL: print('- move -')
        if (self.signal == Field.EXIT): self.finished = True
        else:
            greatest_fields = np.argwhere(
                self.board >= np.amax(self.board) * PERCENTAGE_OF_MOST_PROBABLE_POINTS
                )
            # greatest_fields = np.argwhere(self.board > 0) 
            for field in greatest_fields:
                field_path = self.find_path_to_exit(tuple(field))
                # update
                # field_prob = self.board[field[0], field[1]] 
                for key in self.path.keys(): self.path[key] += field_path[key] # * field_prob
                # print(field, " - > ", field_path)
            
            # if PRINT_ALL: print(self.path)
            # sort dict
            m = {k: v for k, v in sorted(self.path.items(), key=lambda item: item[1])}

            # select best move, get next one if it's counter move
            move = list(m.keys())[-1]
            if self.block_counter_move(move):
                move = list(m.keys())[-2]

            # zeroing dict
            self.path = {key: 0 for key in self.path}
            return move

    # find manhatan distance for PARTICULAR field
    def find_path_to_exit(self, field):
        path = {Action.LEFT : 0, Action.DOWN : 0, Action.RIGHT : 0, Action.UP : 0}
        y, x = field
        y_distance = y - self.exit_loc[0]
        x_distance = x - self.exit_loc[1]
        #
        # distance when goes through LEFT/RIGHT borders
        if self.exit_border_w[0] == 'RIGHT': # distance from left border
            x_distance_2 = x + self.exit_border_w[1]

        elif self.exit_border_w[0] == 'LEFT':# distance from right border
            x_distance_2 = -1*(abs(self.w - x) + self.exit_border_w[1])

        # check if go straight or through borders
        if abs(x_distance_2) < abs(x_distance): x_distance = x_distance_2

        if x_distance > 0:
            path[Action.LEFT] = x_distance
        elif x_distance < 0:
            path[Action.RIGHT] = abs(x_distance)
        else: 
            # do nothnig
            pass
        #
        # distance when goes through UP/DOWN borders
        if self.exit_border_h[0] == 'DOWN': # distance from upper border
            y_distance_2 = y + self.exit_border_h[1]

        elif self.exit_border_h[0] == 'UP':# distance from bottom border
            y_distance_2 = -1*(abs(self.h - y) + self.exit_border_h[1])

        # check if go straight or through borders
        if abs(y_distance_2) < abs(y_distance): y_distance = y_distance_2

        if y_distance > 0:
            path[Action.UP] = y_distance
        elif y_distance < 0:
            path[Action.DOWN] = abs(y_distance)
        else: 
            # do nothnig
            pass

        return path

    # if going up - move probs up etc.
    def update_probabilities(self, move):
        boards_copy = self.board.copy()
        
        if move == Action.UP:
            for height in range(self.h):
                if height == self.h-1:  
                    self.board[self.h-1] = boards_copy[0]
                else: 
                    self.board[height] = boards_copy[height+1]

        elif move == Action.DOWN:
            for height in range(self.h):
                if height == 0:         
                    self.board[0] = boards_copy[self.h-1]
                else: 
                    self.board[height] = boards_copy[height-1]
        
        elif move == Action.RIGHT:
            for width in range(self.w):
                if width == 0:         
                    self.board[:,0] = boards_copy[:,self.w-1]
                else: 
                    self.board[:,width] = boards_copy[:,width-1]
        
        elif move == Action.LEFT:
            for width in range(self.w):
                if width == self.w-1:         
                    self.board[:,self.w-1] = boards_copy[:,0]
                else: 
                    self.board[:,width] = boards_copy[:,width+1]

    # if going up - move probs up etc.
    def update_2(self, move):
        boards_copy = self.board.copy()
        
        if move == Action.UP:
            top_row = boards_copy[0]
            self.board[:-1] = boards_copy[1:]
            self.board[self.h-1] = top_row

        elif move == Action.DOWN:
            bottom_row = boards_copy[self.h-1]
            self.board[1:] = boards_copy[:-1]
            self.board[0] = bottom_row
        
        elif move == Action.RIGHT:
            right_row = boards_copy[:,self.w-1]
            self.board[:,1:] = boards_copy[:,:-1]
            self.board[:,0] = right_row

        elif move == Action.LEFT:
            left_row = boards_copy[:,0]
            self.board[:,:-1] = boards_copy[:,1:]
            self.board[:,self.w-1] = left_row

    # multiply field and its neighbours by p or p_rest
    def recalulate_field(self, i, j, boards_copy):
        # destination
        self.board[i,j]             += boards_copy[i,j]*self.p
        # down
        if i+1 >= self.h:
            self.board[0,j]         += boards_copy[i,j]*self.p_rest
        else: 
            self.board[i+1,j]       += boards_copy[i,j]*self.p_rest
        # up
        if i-1 < 0:
            self.board[self.h-1,j]  += boards_copy[i,j]*self.p_rest
        else: 
            self.board[i-1,j]       += boards_copy[i,j]*self.p_rest
        # right
        if j+1 >= self.w:
            self.board[i,0]         += boards_copy[i,j]*self.p_rest
        else: 
            self.board[i,j+1]       += boards_copy[i,j]*self.p_rest
        # left
        if j-1 < 0:
            self.board[i, self.w-1] += boards_copy[i,j]*self.p_rest
        else: 
            self.board[i,j-1]       += boards_copy[i,j]*self.p_rest      

    # multiply field and its neighbours by 1/len(possible)
    def recalulate_field_probs(self, i, j, prob):
        # destination
        self.board[i,j]             += prob*self.p
        # down
        if i+1 >= self.h:
            self.board[0,j]         += prob*self.p_rest
        else: 
            self.board[i+1,j]       += prob*self.p_rest
        # up
        if i-1 < 0:
            self.board[self.h-1,j]  += prob*self.p_rest
        else: 
            self.board[i-1,j]       += prob*self.p_rest
        # right
        if j+1 >= self.w:
            self.board[i,0]         += prob*self.p_rest
        else: 
            self.board[i,j+1]       += prob*self.p_rest
        # left
        if j-1 < 0:
            self.board[i, self.w-1] += prob*self.p_rest
        else: 
            self.board[i,j-1]       += prob*self.p_rest    

        
    # multiply WHOLE BOARD by p or p_rest
    def recalculate_probabilities(self):
        boards_copy = self.board.copy()
        self.board = np.zeros((self.h, self.w))

        for i in range(self.h):
            for j in range(self.w):
                self.recalulate_field(i,j,boards_copy)

    # update only possible ends of paths
    def recalculate_only_possible_fields(self, possible):
        # self.board = np.zeros((self.h, self.w))
        for field in possible:
            self.recalulate_field_probs(field[0], field[1], 1/len(possible))

    # find all fields which are the ENDPOINTS
    # of already visited path on board
    def find_path_in_map(self):
        possible = []
        for i in range(self.h):
            for j in range(self.w):
                start_point = [i, j]
                add = True
                for point in self.my_path:
                    start_point[0] += point[1] # h
                    start_point[1] += point[2] # w

                    # if height out of bound
                    if start_point[0] >= self.h:
                        start_point[0] = start_point[0]-self.h
                    elif start_point[0] < 0:
                        start_point[0] = self.h + start_point[0]
                    # if width out of bound
                    if start_point[1] >= self.w:
                        start_point[1] = start_point[1]-self.w
                    elif start_point[1] < 0:
                        start_point[1] = self.w + start_point[1]  

                    if self.map[start_point[0], start_point[1]] != point[0]:
                        add = False
                        break

                if add: possible.append(start_point)

        return possible

    def append_to_my_path(self, signal, move):
        if move == Action.UP:
            mh, mw = 1, 0
        elif move == Action.DOWN:
            mh, mw = -1, 0
        elif move == Action.RIGHT:
            mh, mw = 0, 1
        elif move == Action.LEFT:
            mh, mw = 0, -1
        else:
            mh, mw = 0, 0

        self.my_path.append((signal, mh, mw))
        if len(self.my_path) > MY_PATH_MEMORY:
            self.my_path.pop(0)

    def normalize(self):
        p_sum = 0
        for row in self.board:
            p_sum += sum(row)

        if p_sum > 0:
            for i in range(self.h):
                for j in range(self.w):
                    self.board[i,j] /= p_sum

    def random_move(self):
        if PRINT_ALL: print('- random move -')
        if len(self.random_moves) % 2 == 0: 
            move = Action.UP
        else:
            if random.random() > 0.5: move = Action.LEFT
            else: move = Action.RIGHT

        self.random_moves.append(move)
        return move

    # agent.move()
    def move(self):
        """
        1. recalulate_field() po wyborze ruchu w zadanym kierunku
        2. recalculate_only_possible_fields:
            CAVE - 1/possible pomnożyć całość przez pj i pn
            EMPTY- 1/possible tylko dla EMPTY
        """
        # if PRINT_ALL: print('- normal move -')
        # if PRINT_ALL: print(Field(self.signal))
        # self.signal = int(input())

        # append last move to path
        if self.first:
            self.board = np.zeros((self.h, self.w))
            self.first = False
        else:
            pass

        # find all possible sub-path endpoints        
        possible = np.argwhere(self.map == self.signal) 

        # recalucalate probs
        # self.recalculate_only_possible_fields(possible)

        if self.signal == Field.EMPTY:
            probs = 1/len(possible)
            for f in possible:
                self.board[f[0],f[1]] += probs
        
        elif self.signal == Field.CAVE:
            probs = 1/len(possible)
            for f in possible:
                self.board[f[0],f[1]] += probs
                self.board[f[0],f[1]] *= self.pj

            others = np.argwhere(self.map == Field.EMPTY) 
            for f in others:
                self.board[f[0],f[1]] *= self.pn

        self.normalize()
        # if PRINT_ALL: print_result(self.board)  

        # select direction
        self.last_move = self.select_move_direction()
        # print(self.last_move, file=OUTPUT_FILE, flush=True)

        # move probabilities in selected direction
        self.update_2(self.last_move)
        # print_result(self.board)       

        # if PRINT_ALL: print('-- end move --')

        return self.last_move


    def init_move(self):
        if PRINT_ALL: print(self.map)
        # self.board = np.zeros((self.h, self.w))

        # start normal moves
        while not self.finished: 
            self.signal = int(input())
            if (self.signal == Field.EXIT): 
                self.finished = True          
            else:
                self.move()
            

        if PRINT_ALL: print(' -- THE END -- ')



def print_result(lines):
    result = "\n".join(["\t".join([format(x,PREC) for x in line]) for line in lines])
    print(result)


def load_all_from_stdin():
    input_file = sys.stdin
    output_file = sys.stdout
    num_of_worlds = int(input_file.readline())
    for _ in range(num_of_worlds):
        world, p, pj, pn = load_world_from_stdin()
        # do stuff


def load_all_from_file():
    input_file = "tests/2020.in" # del
    worlds = load_input_file(input_file)
    for i in worlds:
        world, p, pj, pn = i
        wumpus = OptWmp(world, p, pj, pn)
        wumpus.reset()
        wumpus.init_move()
        # do stuff


if __name__ == "__main__":
    # load_all_from_file()
    run_agent(OptWmp)


        
