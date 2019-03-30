import sys
from contextlib import closing
from six import StringIO
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

taxi_map = mg.Map()
taxi_map.map_creation(x_size, y_size, z_size)
d = taxi_map.loc_creation()
print(d)
# taxi_map.printmap(-1)

# RGBY: lay-row-column
# for key in d:
#     print(key, d[key])
colors = ['R', 'G', 'Y', 'B']

# print(taxi_map.shape())
# print(taxi_map.volume())

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
        print(self.desc)

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
                                reward = -1  # default reward when there is no pickup/dropoff
                                done = False
                                taxi_loc = (lay, row, col)
                                # 0 - south
                                if action == 0:
                                    new_row = min(row + 1, max_row)
                                # 1 - north
                                elif action == 1:
                                    new_row = max(row - 1, 0)
                                # 2 - east
                                if action == 2 and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, max_col)
                                # 3 - west
                                elif action == 3 and self.desc[lay, 1 + row, 2 * col] == b":":
                                    new_col = max(col - 1, 0)
                                # 4 - move up
                                if action == 4:  # and self.desc[lay, 1 + row, 2 * col + 2] == b":":
                                    new_lay = min(lay + 1, max_lay)
                                # 5 - move down
                                elif action == 5:  # and self.desc[lay, 1 + row, 2 * col] == b":":
                                    new_lay = max(lay - 1, 0)
                                # 6 - pickup
                                elif action == 6:  # pickup
                                    if (pass_idx < 4) and (taxi_loc == self.locs[pass_idx]):
                                        new_pass_idx = 4
                                    else:  # passenger not at location
                                        reward = -10
                                # 7 - dropoff
                                elif action == 7:  # dropoff
                                    if (taxi_loc == self.locs[dest_idx]) and pass_idx == 4:
                                        new_pass_idx = dest_idx
                                        done = True
                                        reward = 20
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
        i = taxi_row
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
        print(out, i)
        i = i // 4
        out.append(i % 5)
        print(out, i)
        i = i // 5
        out.append(i % 5)
        print(out, i)
        i = i // 5
        # Next 3 lines... I'm not really sure
        out.append(i % 5)
        print(out, i)
        i = i // 5
        out.append(i)
        print(out, i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[[c.decode('utf-8') for c in line] for line in lay] for lay in out]
        # debug
        #count = 0
        #for item in out:
        #    print(count)
        #    count = count + 1
        #    for val in item:
        #        print(val)
        
        # self.s ? wtf?
        # self.s defined in discrete.py
        print("self.s", self.s)
        taxi_lay, taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x): return "_" if x == " " else x
        print("self.locs", self.locs)
        # taxi_lay == 0 ALWAYS!!!! WE NEED TO REWRITE DECODE() LOOK AT LINE: 205
        print("taxi_lay", taxi_lay)
        print("pass_idx", pass_idx)
        print("dest_idx", dest_idx)
        print("self.locs[pass_idx]: lay, row, column", self.locs[pass_idx])
        print("self.locs[dest_idx]: lay, row, column", self.locs[dest_idx])
        if pass_idx < 4:
            # necessary to rewrite to 3D, taxi_lay + ... (below)
            out[taxi_lay][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[taxi_lay][1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            # necessary to rewrite to 3D, pk + ... (below)
            pk, pj, pi = self.locs[pass_idx]
            print("pk", pk)
            out[pk][1 + pi][2 * pj + 1] = utils.colorize(out[pk][1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
        # necessary to rewrite to 3D, taxi_lay + ... (below)
            out[taxi_lay][1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[taxi_lay][1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        dk, di, dj = self.locs[dest_idx]
        print("dk", dk)
        # necessary to rewrite to 3D, dk + ... (below)
        out[dk][1 + di][2 * dj + 1] = utils.colorize(out[dk][1 + di][2 * dj + 1], 'magenta')
        # necessary to rewrite to 3D (below)
        #outfile.write("\n".join(["".join(row) for row in out[taxi_lay]]) + "\n")
        outfile.write("\n".join(["".join(["".join(row) for row in lay]) for lay in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "MoveUp", "MoveDown", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

print('taxi.py launched successfully')
