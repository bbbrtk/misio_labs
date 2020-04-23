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


class MyWumpus(object):
    def __init__(self, map: np.ndarray, p: float, pj: float, pn: float, exit_loc: tuple = None):
        self.map = map
        self.board = None
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
        self.finished = True

        self.signal = Field.EMPTY
        self.last_move = Action.RIGHT
        self.my_path = []
        self.first = True
        self.empties = np.argwhere(map == Field.EMPTY)
        self.caves = np.argwhere(map == Field.CAVE)
        self.emp_probs = 0
        self.cav_probs = 0
        self.more_empties = True

    def __str__(self):
        return super().__str__()

    def reset(self):
        self.moves = 0
        self.finished = False

    def sense(self, sensory_input: bool):
        self.signal = sensory_input


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
        the_path = {Action.LEFT : 0, Action.DOWN : 0, Action.RIGHT : 0,  Action.UP : 0}
        greatest_fields = np.argwhere(self.board >= np.amax(self.board))

        for field in greatest_fields:
            field_path = self.find_path_to_exit(tuple(field))
            # field_path = self.path_to_exit_dict[tuple(field)]
            # update
            # field_prob = self.board[field[0], field[1]] 
            for key in the_path.keys(): the_path[key] += field_path[key] # * field_prob
            # print(field, " - > ", field_path)
        
        # if PRINT_ALL: print(the_path)
        # sort dict
        m = {k: v for k, v in sorted(the_path.items(), key=lambda item: item[1])}

        # select best move, get next one if it's counter move
        move = list(m.keys())[-1]
        if self.block_counter_move(move):
            move = list(m.keys())[-2]

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
    def recalulate_field(self, i, j, b_value):
        ij_p_rest = b_value*self.p_rest
        # destination
        self.board[i,j]             += b_value*self.p
        # down
        if i+1 >= self.h:
            self.board[0,j]         += ij_p_rest
        else: 
            self.board[i+1,j]       += ij_p_rest
        # up
        if i-1 < 0:
            self.board[self.h-1,j]  += ij_p_rest
        else: 
            self.board[i-1,j]       += ij_p_rest
        # right
        if j+1 >= self.w:
            self.board[i,0]         += ij_p_rest
        else: 
            self.board[i,j+1]       += ij_p_rest
        # left
        if j-1 < 0:
            self.board[i, self.w-1] += ij_p_rest
        else: 
            self.board[i,j-1]       += ij_p_rest   

    # ?
    def recalulate_field_2(self, i, j):
        # destination
        self.board[i,j]             *= self.p
        # down
        if i+1 >= self.h:
            self.board[0,j]         *= self.p_rest
        else: 
            self.board[i+1,j]       *= self.p_rest
        # up
        if i-1 < 0:
            self.board[self.h-1,j]  *= self.p_rest
        else: 
            self.board[i-1,j]       *= self.p_rest
        # right
        if j+1 >= self.w:
            self.board[i,0]         *= self.p_rest
        else: 
            self.board[i,j+1]       *= self.p_rest
        # left
        if j-1 < 0:
            self.board[i, self.w-1] *= self.p_rest
        else: 
            self.board[i,j-1]       *= self.p_rest   
     
    # multiply WHOLE BOARD by p or p_rest
    def recalculate_probabilities(self):
        boards_copy = self.board.copy()
        self.board = np.zeros((self.h, self.w))

        for i in range(self.h):
            for j in range(self.w):
                self.recalulate_field(i,j,boards_copy)

    # !!!
    def recalulate(self):
        if self.signal == Field.EMPTY:
            if self.more_empties:
                self.board += self.emp_probs
                for f in self.caves:
                    self.board[f[0],f[1]] -= self.emp_probs
            else:
                for f in self.empties:
                    self.board[f[0],f[1]] += self.emp_probs

        
        elif self.signal == Field.CAVE:
            if not self.more_empties:
                self.board += self.cav_probs
                self.board *= self.pj
                for f in self.empties:
                    self.board[f[0],f[1]] = ((self.board[f[0],f[1]]/self.pj) - self.cav_probs) * self.pn

            else:
                self.board *= self.pn
                for f in self.caves:
                    self.board[f[0],f[1]] = (self.board[f[0],f[1]]/self.pn + self.cav_probs) * self.pj



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
            mh, mw = -1, 0
        elif move == Action.DOWN:
            mh, mw = 1, 0
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
            self.board = self.board / p_sum


    # agent.move()
    def move(self):

        # if PRINT_ALL: print('- normal move -')
        # if PRINT_ALL: print(Field(self.signal))
        # self.signal = int(input())

        # append last move to path
        if self.first:
            self.first = False
            self.board = np.zeros((self.h, self.w))
            self.emp_probs = 1/len(self.empties)
            self.cav_probs = 1/len(self.caves)
            if self.emp_probs > self.cav_probs: 
                self.more_empties = False
        else:
            greatest_fields = np.argwhere(self.board >= np.amax(self.board))
            for i in greatest_fields:
                self.recalulate_field(i[0],i[1],self.board[i[0],i[1]])

        self.recalulate()
        self.normalize()
        self.board[self.exit_loc[0],self.exit_loc[1]] = 0
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
        wumpus = MyWumpus(world, p, pj, pn)
        wumpus.reset()
        wumpus.init_move()
        # do stuff


if __name__ == "__main__":
    # load_all_from_file()
    run_agent(MyWumpus)




