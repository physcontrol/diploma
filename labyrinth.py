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

MAP_default = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

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
for a in range(z_size-1,-1,-1):
    lay_reward[a] = counter
    counter = counter - 1
print("LAY REWARD: ", lay_reward)
time.sleep(1)

cell_reward = np.zeros((x_size, y_size, z_size))
for z, lay in enumerate(cell_reward):
    #we need to rewrite this part for getting more paramters( Density and Wind, for example)
    struct = {
        'Lay Reward': lay_reward[z],
        'Density': 0,
        'Pressure': 0,
        'Wind': 0,
        }
    for y, col in enumerate(lay):
        for x, row in enumerate(col):
            result = functional(struct)
            cell_reward[z][y][x] = result
            #print(x,y,z, row)
print(cell_reward)
time.sleep(4)

taxi_map = mg.Map()
taxi_map.map_creation(x_size, y_size, z_size)
d = taxi_map.loc_creation()
print(d)
#taxi_map.printmap(-1)

# RGBY: lay-row-column
# for key in d:
#     print(key, d[key])
colors = ['R', 'G', 'Y', 'B']

# print(taxi_map.shape())
# print(taxi_map.volume())
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
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    Description:
    There are four designated locations in the grid world indicated by R(ed), B(lue), G(reen), and Y(ellow).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxi drive to the passenger's location, pick up the passenger, drive to the passenger's destination
    (another one of the four specified locations), and then drop off the passenger. Once the passenger is dropped off,
    the episode ends.

    Observations: 
    
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger
    (including the case when the passenger is the taxi), and 4 destination locations.
    
    Actions: 
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west
    - 4: move up
    - 5: move down
    - 6: pickup passenger
    - 7: dropoff passenger


    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delievering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.
    

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, B and Y): locations for passengers and destinations

    actions:
    - 0: south
    - 1: north
    - 2: east
    - 3: west
    - 4: up
    - 5: down
    - 6: pickup
    - 7: dropoff

    state space is represented by:
        (taxi_lay, taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        #print(taxi_map.printmap(3))
        self.desc = np.asarray(taxi_map.MAP, dtype='c')
        # self.desc = np.ndarray(shape=taxi_map.shape(), dtype='c', buffer=taxi_map.MAP)
        # print(self.desc)
        self.locs = []

        for c in colors:
            self.locs.append(d[c])
        #print(self.desc)

        # num_states for 3D, 2D is 5 times lower
        num_states = taxi_map.volume()*5*4
        num_rows = y_size
        num_columns = x_size
        num_lays = z_size
        # Do we really need these rows?
        # yes!
        max_row = num_rows - 1
        max_col = num_columns - 1
        max_lay = num_lays - 1
        # I feel that we should use 3D matrix here instead of 2D one
        # May be, i will think about it...
        initial_state_distrib = np.zeros(num_states)
        # pickUp, dropOff, NEW and Up, Down == 8
        num_actions = 8
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        # Running through layers (Z-axis)
        for lay in range(num_lays):
            # Running through rows (Y-axis)
            for row in range(num_rows):
                # Running through columns (X-axis)
                for col in range(num_columns):
                    for pass_idx in range(len(self.locs) + 1):  # +1 for being inside taxi
                        for dest_idx in range(len(self.locs)):
                            state = self.encode(lay, row, col, pass_idx, dest_idx)
                            if pass_idx < 4 and pass_idx != dest_idx:
                                initial_state_distrib[state] += 1
                            for action in range(num_actions):
                                # defaults
                                new_lay, new_row, new_col, new_pass_idx = lay, row, col, pass_idx
                                # lay_reward is hashmap, which created above
                                #reward = lay_reward[lay]
                                ### default reward when there is no pickup/dropoff
                                #reward = -1
                                reward = cell_reward[lay][col][row]
                                done = False
                                taxi_loc = (lay, row, col)
                                # 0 - south
                                if action == 0:
                                    #old version
                                    #new_row = min(row + 1, max_row)
                                    #new version
                                    new_row, boom = get_minimum(row + 1, max_row)
                                    if boom:
                                        reward = -10
                                # 1 - north
                                elif action == 1:
                                    #old version
                                    #new_row = max(row - 1, 0)
                                    #new version
                                    new_row, boom = get_maximum(row - 1, 0)
                                    if boom:
                                        reward = -10
                                # 2 - east
                                if action == 2 #and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    #old version
                                    #new_col = min(col + 1, max_col)
                                    #new version
                                    new_col, boom = get_minimum(col + 1, max_col)
                                    if boom:
                                        reward = -10
                                # 3 - west
                                elif action == 3 #and self.desc[lay, 1 + row, 2 * col] == b":":
                                    #old version
                                    #new_col = max(col - 1, 0)
                                    #new version
                                    new_col, boom = get_maximum(col - 1, 0)
                                    if boom:
                                        reward = -10
                                # 4 - move up
                                if action == 4:  # and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    #old ersion
                                    #new_lay = min(lay + 1, max_lay)
                                    #new ersion
                                    new_lay, boom = get_minimum(lay + 1, max_lay)
                                    if boom:
                                        reward = -10
                                # 5 - move down
                                elif action == 5:  # and self.desc[lay, 1 + row, 2 * col] == b":":
                                    #old version
                                    #new_lay = max(lay - 1, 0)
                                    #new version
                                    new_lay, boom = get_maximum(lay - 1, 0)
                                    if boom:
                                        reward = -10
                                # 6 - pickup
                                elif action == 6:  # pickup
                                    if (pass_idx < 4) and (taxi_loc == self.locs[pass_idx]):
                                        #print("passenger not at location")
                                        new_pass_idx = 4
                                    else:  # passenger not at location
                                        #print("passenger not at location")
                                        reward = -10
                                # 7 - dropoff
                                elif action == 7:  # dropoff
                                    if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                        new_pass_idx = dest_idx
                                        done = True
                                        reward = 30
                                    elif (taxi_loc in self.locs) and pass_idx == 4:
                                        new_pass_idx = self.locs.index(taxi_loc)
                                    else:  # dropoff at wrong location
                                        reward = -10
                                new_state = self.encode(
                                    new_lay, new_row, new_col, new_pass_idx, dest_idx)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_lay, taxi_row, taxi_col, pass_loc, dest_idx):
        # (5) 5, 5, 4
        i = taxi_lay
        i *= 5
        i += taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[[c.decode('utf-8') for c in line] for line in lay] for lay in out]
        taxi_lay, taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        '''
        print("taxi_row", taxi_row)
        print("taxi_col", taxi_col)
        print("taxi_lay", taxi_lay)
        print("pass_idx", pass_idx)
        print("dest_idx", dest_idx)
        print("self.locs[pass_idx]: lay, row, column", self.locs[pass_idx])
        print("self.locs[dest_idx]: lay, row, column", self.locs[dest_idx])
        '''
        if pass_idx < 4:
            # necessary to rewrite to 3D, taxi_lay + ... (below)
            out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            # necessary to rewrite to 3D, pk + ... (below)
            pk, pj, pi = self.locs[pass_idx]
            '''
            print("pk", pk)
            print("pi", pi)
            print("pj", pj)
            '''
            #pj is columns, we need use 2*pj + 1(because we have ":")
            #but pk, pi values in [1,5]
            out[pk + 1][pi + 1][2 * pj + 1] = utils.colorize(out[pk + 1][pi + 1][2 * pj + 1], 'blue', bold=True)
            #old version below
            #out[pk + 1][1 + pi][2 * pj + 1] = utils.colorize(out[pk + 1][1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
        # necessary to rewrite to 3D, taxi_lay + ... (below)
            out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[taxi_lay + 1][1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        dk, di, dj = self.locs[dest_idx]
        '''
        print("dk", dk)
        print("di", di)
        print("dj", dj)
        '''
        # necessary to rewrite to 3D, dk + ... (below)
        #dj is columns, we need use 2*pj + 1(because we have ":")
        #but dk, di values in [1,5]
        out[dk + 1][di + 1][2 * dj + 1] = utils.colorize(out[dk + 1][di + 1][2 * dj + 1], 'magenta')
        #old version below
        #out[dk][1 + di][2 * dj + 1] = utils.colorize(out[dk][1 + di][2 * dj + 1], 'magenta')
        # necessary to rewrite to 3D (below)
        #outfile.write("\n".join(["".join(row) for row in out[taxi_lay]]) + "\n")
        #outfile.write("\n".join(["".join(["".join(row) for row in lay]) for lay in out]) + "\n")
        #print only lay with taxi
        time.sleep(0.5)
        os.system('clear')
        print("TAXI:")
        print("LAY: ", taxi_lay)
        print("ROW: ", taxi_row)
        print("COLUMN: ", taxi_col)
        outfile.write("\n".join(["".join(row) for row in out[taxi_lay + 1]]) + "\n")
        # print all lays below
        #print("ALL LAYS")
        #for item in out:
        #    outfile.write("\n".join(["".join(row) for row in item]) + "\n")
        #print("SELF_LASTACTION: ", self.lastaction)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "MoveUp", "MoveDown", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

print('taxi.py launched successfully')



