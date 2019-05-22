import sys
from contextlib import closing
from six import StringIO
import os
import time
from gym import utils
from gym.envs.toy_text import discrete
#from gym.envs.toy_text import map_generation as mg
import map_generation as mg
import numpy as np

x_size = 5
y_size = x_size
z_size = x_size

#functional that sets of paramenters in accordance with reward
def functional(inPut):
    out = 0
    for val in inPut:
        out += inPut[val]
    return out

lay_reward = {}
counter = -1
fine_step = 0.5
for a in range(z_size-1,-1,-1):
    lay_reward[a] = counter
    counter = counter - fine_step

cell_reward = np.zeros((x_size, y_size, z_size))
for z, lay in enumerate(cell_reward):
    struct = {
        'Lay Reward': lay_reward[z],
        'Density': 0,
        'Wind': 0,
        }
    for y, col in enumerate(lay):
        for x, row in enumerate(col):
            result = functional(struct)
            cell_reward[z][y][x] = result

taxi_map = mg.Map()
taxi_map.map_creation(x_size, y_size, z_size)
d = taxi_map.loc_creation()

colors = ['R', 'G', 'Y', 'B']

def get_minimum(new_location, limit):
    boom = False
    if new_location > limit:
        return limit, True
    return new_location, boom

def get_maximum(new_location, limit):
    boom = False
    if new_location < limit:
        return limit, True
    return new_location, boom
    
class TaxiEnv(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(taxi_map.MAP, dtype='c')
        self.locs = []

        for c in colors:
            self.locs.append(d[c])

        num_states = taxi_map.volume()*5*4
        num_rows = y_size
        num_columns = x_size
        num_lays = z_size
        max_row = num_rows - 1
        max_col = num_columns - 1
        max_lay = num_lays - 1
        
        initial_state_distrib = np.zeros(num_states)
        num_actions = 8
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        # Running through layers (Z-axis)
        for lay in range(num_lays):
            # Running through rows (Y-axis)
            for row in range(num_rows):
                # Running through columns (X-axis)
                for col in range(num_columns):
                    for obj_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                        for dest_idx in range(len(self.locs)):
                            state = self.encode(lay, row, col, obj_idx, dest_idx)
                            if obj_idx < 4 and obj_idx != dest_idx:
                                initial_state_distrib[state] += 1
                            for action in range(num_actions):
                                new_lay, new_row, new_col, new_obj_idx = lay, row, col, obj_idx
                                #reward = lay_reward[lay]
                                reward = cell_reward[lay][col][row]
                                done = False
                                taxi_loc = (lay, row, col)
                                # 0 - south
                                if action == 0:
                                    new_row, boom = get_minimum(row + 1, max_row)
                                    if boom:
                                        reward = -10
                                # 1 - north
                                elif action == 1:
                                    new_row, boom = get_maximum(row - 1, 0)
                                    if boom:
                                        reward = -10
                                # 2 - east
                                if action == 2 and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    new_col, boom = get_minimum(col + 1, max_col)
                                    if boom:
                                        reward = -10
                                # 3 - west
                                elif action == 3 and self.desc[lay, 1 + row, 2 * col] == b":":
                                    new_col, boom = get_maximum(col - 1, 0)
                                    if boom:
                                        reward = -10
                                # 4 - move up
                                if action == 4:  # and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    new_lay, boom = get_minimum(lay + 1, max_lay)
                                    if boom:
                                        reward = -10
                                # 5 - move down
                                elif action == 5:  # and self.desc[lay, 1 + row, 2 * col] == b":":
                                    new_lay, boom = get_maximum(lay - 1, 0)
                                    if boom:
                                        reward = -10
                                # 6 - pickup
                                elif action == 6:  # pickup
                                    if (obj_idx < 4) and (taxi_loc == self.locs[obj_idx]):
                                        new_obj_idx = 4
                                    else:  #object not at location
                                        reward = -10
                                # 7 - dropoff
                                elif action == 7:  # dropoff
                                    if (taxi_loc == self.locs[dest_idx]) and obj_idx == 4:
                                        new_obj_idx = dest_idx
                                        done = True
                                        reward = 50
                                    elif (taxi_loc in self.locs) and obj_idx == 4:
                                        new_obj_idx = self.locs.index(taxi_loc)
                                    else:  # dropoff at wrong location
                                        reward = -20
                                new_state = self.encode(
                                    new_lay, new_row, new_col, new_obj_idx, dest_idx)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_lay, taxi_row, taxi_col, obj_loc, dest_idx):
        i = taxi_lay
        i *= z_size
        i += taxi_row
        i *= y_size
        i += taxi_col
        i *= x_size
        i += obj_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % x_size)
        i = i // x_size
        out.append(i % y_size)
        i = i // y_size
        out.append(i % z_size)
        i = i // z_size
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[[c.decode('utf-8') for c in line] for line in lay] for lay in out]
        taxi_lay, taxi_row, taxi_col, obj_idx, dest_idx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if obj_idx < 4:
            out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pk, pj, pi = self.locs[obj_idx]
            out[pk + 1][pi + 1][2 * pj + 1] = utils.colorize(out[pk + 1][pi + 1][2 * pj + 1], 'blue', bold=True)
        else:  #agent with object
            out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        dk, di, dj = self.locs[dest_idx]
        out[dk + 1][di + 1][2 * dj + 1] = utils.colorize(out[dk + 1][di + 1][2 * dj + 1], 'magenta')
        os.system('clear')
        print("AGENT:")
        print("LAY: ", taxi_lay)
        print("ROW: ", taxi_row)
        print("COLUMN: ", taxi_col)
        outfile.write("\n".join(["".join(row) for row in out[taxi_lay + 1]]) + "\n")
        time.sleep(5)
        #print all lays below
        #print("ALL LAYS")
        #for item in out:
        #    outfile.write("\n".join(["".join(row) for row in item]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "MoveUp", "MoveDown", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

print('labyrinth.py launched successfully')
