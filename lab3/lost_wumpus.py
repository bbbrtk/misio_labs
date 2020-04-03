#!/usr/bin/python3

from misio.lost_wumpus._wumpus import *
from misio.lost_wumpus.util import load_input_file, load_world_from_stdin
import numpy as np
import itertools, time  
from itertools import chain
from operator import itemgetter
import signal

PREC = '.8f'

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
        if exit_loc is None:
            exit_loc = [int(x) for x in np.where(map == Field.EXIT)]
        self.exit_loc = list(exit_loc)
        self.position = list(exit_loc)
        self.moves = np.inf
        self.finished = True
        self.sensory_output = None
        if max_moves is None:
            max_moves = np.inf
        self.max_moves = max_moves

        self.signal = self.position

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
                if self.signal == Field.CAVE:
                    if self.map[i,j] == Field.CAVE:     self.board[i,j] *= self.pj
                    elif self.map[i,j] == Field.EMPTY:  self.board[i,j] *= self.pn
                    else: self.board[i,j] = 0


        greatest_fields = np.argwhere(self.board == np.amax(self.board))
        print(greatest_fields)

        print_result(self.board)

    def move(self):
        print(Field(self.signal))
        """ TODO
        0. if dont know where we are - do random moves but without returns:
            up
            left or right
            up
            left or right
            and so on

        1.  take ~10% of most probable ones (round down if less than 10% are unique):
            1.1. greatest_fields = np.argwhere(self.board == np.amax(self.board))
            1.2 second_greatest = np.argwhere(self.board >= np.amax(self.board) - some_number)
            take first quartille or something

        2. for each great_field find path to exit and store as a dict of moves {'left': 2, 'up': 1}

        3. make move wich occures the most often in all great_fields

        repeat
        
        """

        if (self.signal == Field.EXIT): self.finished = True

        print("-- end move --")


    def go(self):
        print(self.map)
        while not self.finished:
            self.signal = int(input())
            self.first_analyze()
            self.move()

        print("-- THE END --")



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
    import sys 
    load_all_from_file()


        

