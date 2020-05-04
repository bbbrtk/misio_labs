#!/bin/bash
from scipy.signal import convolve2d
from misio.optilio.lost_wumpus import run_agent
from misio.lost_wumpus.agents import SnakeAgent, AgentStub, RandomAgent
from misio.lost_wumpus._wumpus import Action, Field
import numpy as np

MY_PATH_MEMORY = 3

# [0,0] UP => [4,0]
# [0,0] LEFT => [0,4]
def safe_position(x: int, boundary: int):
    return (boundary + x) % boundary

# returns position of the field 
# after performing action from field current_position
def cyclic_position(current_position: (int, int), action: Action, boundary: (int, int)) -> (int, int):
    h, w = boundary
    y, x = current_position
    if action == Action.DOWN:
        y = safe_position(y+1, h)
    elif action == Action.UP:
        y = safe_position(y-1, h)
    elif action == Action.RIGHT:
        x = safe_position(x+1, w)
    else:
        x = safe_position(x-1, w)
    return y, x

# example: x = 1, y = 4, boundary = 5, options (Action.Left, Action.Right)
# should return (2, Action.Left)
def cyclic_distance(src: int, dst: int, boundary: int, options: (Action, Action)) -> (int, Action):
    through_0_distance = min(src, dst) + boundary - max(src, dst)
    distance = abs(dst - src)
    if through_0_distance < distance:
        # it is shorter to go through 0
        # but if src < dst go left, otherwise go right
        return through_0_distance, options[0] if src < dst else options[1]
    else:
        return distance, options[1] if src < dst else options[0]



class Move:
    def __init__(self, location, dy, dx, p, dir):
        self.location = location
        self.dy = dy
        self.dx = dx
        self.p = p
        self.dir = dir

class MyAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)
        self.p_rest = (1-self.p)/4
        self.exit_border_h = ('LEFT', 0)
        self.exit_border_w = ('UP', 0)
        self.exit_loc = 0
        self.exit_loc2 = None
        self.masks = self._make_masks()
        self.cav_mask = np.where(self.map == Field.CAVE, self.pj, self.pn)
        self.emp_mask = np.where(self.map == Field.EMPTY, self.pj, self.pn)
        self.empties = np.argwhere(self.map == Field.EMPTY)
        self.caves = np.argwhere(self.map == Field.CAVE)
        self.emp_probs = 0
        self.cav_probs = 0
        self.more_empties = True
        self.board = []
        self.moves = 0
        self.finished = False
        self.reset()

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

    def sense(self, sensory_input: bool):
        if sensory_input == Field.CAVE:
            self.board *= self.cav_mask
        else:
            self.board *= self.emp_mask

        # self.normalize()
        # self.board /= self.board.max()
        self.board[self.exit_loc] = 0

    def _make_masks(self):
        direction_dict = {}
        defaults = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, self.p_rest, 0],
            [self.p_rest, self.p, self.p_rest],
            [0, self.p_rest, 0]
        ])
        # down = defaults
        right = np.rot90(defaults, 1)
        up = np.rot90(right, 1)
        left = np.rot90(up, 1)

        direction_dict[Action.DOWN] = defaults
        direction_dict[Action.UP] = up
        direction_dict[Action.LEFT] = left
        direction_dict[Action.RIGHT] = right

        return direction_dict

    def select_move_next_direction(self):
        max_val = self.board.max()
        votes = {
            Action.LEFT: 0,
            Action.RIGHT: 0,
            Action.UP: 0,
            Action.DOWN: 0
        }
        for r, row in enumerate(self.board):
            for c, val in enumerate(row):
                if val > max_val * 1:
                    dy, y_direct = cyclic_distance(r, self.exit_loc[0], self.h, (Action.UP, Action.DOWN))
                    dx, x_direct = cyclic_distance(c, self.exit_loc[1], self.w, (Action.LEFT, Action.RIGHT))
                    votes[y_direct] += val**2 + (1 - dy/(self.h*.5))**3
                    votes[x_direct] += val**2 + (1 - dx/(self.w*.5))**3
        return sorted(votes.items(), key=lambda x: x[1], reverse=True)

    def move(self):
        self.moves += 1
        current_move = Action.UP

        # !!!
        list_of_next_moves = self.select_move_next_direction()
        best_move, m = self.select_move_direction()
        second = list(m.keys())[-2]

        if self.moves > 1:
            self.prelast_move = self.last_move

        current_move = best_move
        self.last_move = best_move
        self.the_same_direction_counter = 0

        if current_move in [Action.LEFT, Action.RIGHT] and self.prelast_move in [Action.LEFT, Action.RIGHT]:
            self.returning += 1
            if self.returning >= 2:
                for vote in list_of_next_moves:
                    if vote[0] not in [Action.LEFT, Action.RIGHT]:
                        current_move = vote[0]
                        break
        elif current_move in [Action.DOWN, Action.UP] and self.prelast_move in [Action.DOWN, Action.UP]:
            self.returning += 1
            if self.returning >= 2:
                for vote in list_of_next_moves:
                    if vote[0] not in [Action.DOWN, Action.UP]:
                        current_move = vote[0]
                        break
        else:
            self.returning = 0


        if self.block_counter_move(current_move):
            current_move = second

        self.last_move = current_move

        self.board = convolve2d(self.board, self.masks[current_move], 'same', "wrap")
        self.normalize()

        return current_move

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
            field_path = self._find_path_to_exit(tuple(field))
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

        return move, m

    # find manhatan distance for PARTICULAR field
    def _find_path_to_exit(self, field):
        path = {Action.LEFT : 0, Action.DOWN : 0, Action.RIGHT : 0, Action.UP : 0}
        y, x = field
        y_distance = y - self.exit_loc2[0]
        x_distance = x - self.exit_loc2[1]
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


    def reset(self):
        self.board = np.ones_like(self.map)
        self.moves = 1
        self.exit_loc = np.unravel_index(np.argmax(self.map), self.map.shape)
        self.the_same_direction_counter = 0
        self.moves = 0
        self.finished = False
        if self.exit_loc2 is None:
            ax2 = [int(x) for x in np.where(self.map == Field.EXIT)]
            ey, ex = ax2
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

            self.exit_loc2 = list(ax2)

        self.last_move = None
        self.prelast_move = None
        self.returning = 0


if __name__ == "__main__":
    run_agent(MyAgent)
