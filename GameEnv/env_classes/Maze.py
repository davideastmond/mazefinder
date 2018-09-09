import random as rnd
import numpy as np


class MazeDirection:
    """
    This is a quick reference to orientation, along the outer edges of the game maze
    """
    NORTH = 0
    SOUTH = 2
    WEST = 1
    EAST = 3

class Marker:
    """
    These are the different elements on the game board
    """
    EMPTY = 0
    GOAL = 9
    PLAYER_LOC = 1
    START = 2
    TRAP = 3

class Maze:
    """
     The maze class is hte game board. It will auto-generate a starting position and an ending goal,
     both of which must be on the board edges. Users of this class need to specify the dimensions of the
     board.
    """
    def __init__(self, tuple_size, int_obstacleCount=0):
        """
        :param tuple_size: [rows, cols] the size of the maze board
        :param int_obstacleCount: number of obstacles (traps)
        """
        # The Game board, initialized to a numpy zeros array
        self.grid = np.zeros([tuple_size[0], tuple_size[1]], dtype=int)

        # Private properties for the start location and end location
        self._start_location = ()  # Where the player starts. It is fixed for a given environment
        self._goal_location = ()  # Where the goal is. This is is fixed

        # Empty Space tracker (a list)
        self.empty_Spaces = []

        # Populate the empty spaces list
        self.populate_empty_spaces()
        pass

    @property
    def start_location(self):
        return self._start_location

    @property
    def goal_location(self):
        return self._goal_location

    def populate_empty_spaces(self):
        """""
        basically populates the Empty_Spaces list with tuples of coords that refer to empty spaces
        :return: void
        """

        # Iterate through the grid
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                self.empty_Spaces.append((row, col))

    def create_entry_exit(self):
        """
        places an entry mark and goal mark on the game board.
        :return: void
        """

        """
        First we determine the start position of the player. It must be on the edges (either north, south, west, east)
        We'll use a randomization to randomly pick which one it will be
        """
        rndStartLocation = rnd.randint(0, 4)  # get a random number between 0 and 4 inclusive
        if rndStartLocation == MazeDirection.NORTH:
            # the starting position needs to be on the top edge
            rnd_north_col_position = rnd.randint(0, self.grid.shape[1])
            #  self.grid[0, rnd_north_col_position] ==
            pass
        elif rndStartLocation == MazeDirection.SOUTH:
            pass
        elif rndStartLocation == MazeDirection.WEST:
            pass
        elif rndStartLocation == MazeDirection.EAST:
            pass


a = Maze([5, 5])
print(a.empty_Spaces)
