colors = ['R', 'G', 'Y', 'B']

class Map:
    def __init__(self):
        self.size = []
        self.MAP = []
        self.x_size = 0
        self.y_size = 0
        self.z_size = 0

    def map_creation(self, x_size, y_size, z_size):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size

        for k in range(0, self.z_size+2):
            map_layer = []
            for j in range(0, self.y_size+2):
                new_str = ""
                for i in range(0, 2*self.x_size+1):
                    if (j == 0) or (j == self.y_size+1):
                        if(i == 0) or (i == 2*self.x_size):
                            new_str += "+"
                        else:
                            new_str += "-"
                    if (k == 0) or (k == self.z_size+1):
                        if (j != 0) and (j != self.y_size + 1):
                            if(i == 0) or (i == 2*self.x_size):
                                new_str += "|"
                            else:
                                if (i+1) % 2 == 0:
                                    new_str += "-"
                                else:
                                    new_str += "|"
                    if (k != 0) and (k != self.z_size+1):
                        if (j != 0) and (j != self.y_size + 1):
                            if(i == 0) or (i == 2*self.x_size):
                                new_str += "|"
                            else:
                                if (i+1) % 2 == 0:
                                    new_str += " "
                                else:
                                    new_str += ":"
                map_layer.extend([new_str])
            self.MAP += [map_layer]

    def loc_creation(self):
        row_size = self.y_size
        col_size = self.x_size
        h_size = self.z_size
        count = -1
        randomR = Map.getRandom(self, h_size, row_size, col_size)
        randomG = Map.getRandom(self, h_size, row_size, col_size)
        randomY = Map.getRandom(self, h_size, row_size, col_size)
        randomB = Map.getRandom(self, h_size, row_size, col_size)
        d = dict(R=randomR, G=randomG, B=randomB, Y=randomY)
        for item in self.MAP:
            count = count + 1
            if count == 0 or count == len(self.MAP) - 1:
                continue
            for c in colors:
                lay, row, col = d[c]
                if lay == count:
                    item[row] = item[row][:2*col+1] + c + item[row][2*col+1 + 1:]
        return d

    def getRandom(self, h_size, row_size, col_size):
        import random
        lay = random.randrange(1, h_size, 1)
        row = random.randrange(1, row_size, 1)
        column = random.randrange(1, col_size, 1)
        return lay, row, column

    def printmap(self, layer):
        if layer < 0:
            count = -1
            for item in self.MAP:
                count = count + 1
                print(count)
                for it in item:
                    print(it)
        else:
            for row in range(0, self.y_size+2):
                print(self.MAP[layer][row])

    def shape(self):
        return self.x_size+2, self.y_size+2, self.z_size+2

    def volume(self):
        return self.x_size*self.y_size*self.z_size
