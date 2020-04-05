#!/usr/bin/python3

from misio.lost_wumpus._wumpus import *
from misio.lost_wumpus.util import load_input_file, load_world_from_stdin
import numpy as np
import itertools, time  
from itertools import chain
from operator import itemgetter
import random
import sys

PREC = '.8f'
OUTPUT_FILE = sys.stdout

class Field(IntEnum):
    """Znaki z ktorych sklada sie mapa swiata."""

    EXIT = 2
    """Znak umieszczany mapie oznaczajacy pole z wyjsciem."""

    CAVE = 1
    """Znak umieszczany mapie oznaczajacy pole z jama."""

    EMPTY = 0
    """Znak umieszczany mapie oznaczajacy puste pole."""


class Action(Enum):
    UP = 'UP'
    """Ruch w gore (w kierunku malejacych indeksow Y)."""

    DOWN = 'DOWN'
    """Ruch w dol (w kierunku rosnacych indeksow Y)."""

    LEFT = 'LEFT'
    """Ruch w lewo (w kierunku malejacych indeksow X)."""

    RIGHT = 'RIGHT'
    """Ruch w prawo (w kierunku rosnacych indeksow X)."""


class LostWumpus(object):
    def __init__(self, map: np.ndarray, p: float, pj: float, pn: float, exit_loc: tuple = None, max_moves=None):
        assert isinstance(map, np.ndarray)
        self.map = map
        self.board = map.astype('float64')
        self.h, self.w = map.shape
        self.p = p
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

    def __str__(self):
        return super().__str__()

    def reset(self):
        self.moves = 0
        self.finished = False


    def apply_move(self, action: Action):
        assert not self.finished
        assert isinstance(action, Action)
        motion = [0, 0]
        if action == Action.LEFT:
            motion[1] -= 1
        elif action == Action.RIGHT:
            motion[1] += 1
        elif action == Action.UP:
            motion[0] -= 1
        elif action == Action.DOWN:
            motion[0] += 1

        if np.random.random() > self.p:
            motion[np.random.randint(2)] += np.random.choice([-1, 1])
        self.position[0] += motion[0]
        self.position[1] += motion[1]
        self.position[0] %= self.h
        self.position[1] %= self.w

        self.moves += 1
        self._update_sensory_output()

        if self.position == self.exit_loc or self.moves >= self.max_moves:
            self.finished = True
            self.sensory_output = None

    def _update_sensory_output(self):
        if self.map[self.position[0], self.position[1]] == Field.CAVE:
            self.sensory_output = np.random.binomial(1, self.pj)
        else:
            self.sensory_output = np.random.binomial(1, self.pn)

    def first_analyze(self):
        all_fields = self.h * self.w - 1
        field_prob = 1/all_fields
        
        for i in range(self.h):
            for j in range(self.w):
                self.board[i,j] = float(field_prob)

                if self.map[i,j] == Field.CAVE:     self.board[i,j] *= self.pj
                elif self.map[i,j] == Field.EMPTY:  self.board[i,j] *= self.pn
                else: self.board[i,j] = 0


    def first_move(self):
        print('- first move -')
        if (self.signal == Field.EXIT): self.finished = True
        else:
            greatest_fields = np.argwhere(self.board >= np.amax(self.board)*0.9)
            for field in greatest_fields:
                field_path = self.find_path_to_exit(tuple(field))
                # update
                for key in self.path.keys(): self.path[key] += field_path[key] # multiply by probability

                print(field, " - > ", field_path)
            
            print(self.path)
            move = max(self.path.items(), key=itemgetter(1))[0]
            self.path = {key: 0 for key in self.path}
            return move


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
                    self.board[:][0] = boards_copy[:][self.w-1]
                else: 
                    self.board[:][width] = boards_copy[:][width-1]
        
        elif move == Action.LEFT:
            for width in range(self.w):
                if width == self.w-1:         
                    self.board[:][self.w-1] = boards_copy[:][0]
                else: 
                    self.board[:][width] = boards_copy[:][width+1]

        # exit = 0

    def move(self):
        print('- normal move -')
        print(Field(self.signal))


        """ TODO
        
            0. move probabilities

            1. calculate probability of move

            2. update probabilities in board table
        
        """

        

        print('-- end move --')


    def random_move(self):
        print('- random move -')
        if len(self.random_moves) % 2 == 0: 
            move = Action.UP
        else:
            if random.random() > 0.5: move = Action.LEFT
            else: move = Action.RIGHT

        self.random_moves.append(move)
        return move


    def go(self):
        print(self.map)

        # find first cave to define wumpus position
        self.signal = int(input())
        while self.signal == Field.EMPTY: # as long as we know nothing about position
            move = self.random_move()
            print(move, file=OUTPUT_FILE, flush=True)
            self.signal = int(input())

        # do first move
        self.first_analyze()
        print_result(self.board)
        move = self.first_move()
        print(move, file=OUTPUT_FILE, flush=True)

        self.update_probabilities(move)
        print_result(self.board)

        # start normal moves
        while not self.finished: 
            self.signal = int(input())
            if (self.signal == Field.EXIT): self.finished = True          
            self.move()
            

        print(' -- THE END -- ')



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
        wumpus = LostWumpus(world, p, pj, pn)
        wumpus.reset()
        wumpus.go()
        # do stuff


if __name__ == "__main__":
    load_all_from_file()


        

