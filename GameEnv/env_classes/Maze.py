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
        self.create_entry_exit()
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
        rnd_start_location = rnd.randint(0, 4)  # get a random number between 0 and 4 inclusive
        if rnd_start_location == MazeDirection.NORTH:
            # the starting position needs to be on the top edge
            rnd_north_col_position = rnd.randint(0, self.grid.shape[1] - 1)
            self.grid[0, rnd_north_col_position] = Marker.START
            self._start_location = (0, rnd_north_col_position)
            self.empty_Spaces.remove((0, rnd_north_col_position))

        elif rnd_start_location == MazeDirection.SOUTH:
            # Starting position is the outer bottom edge
            rnd_south_col_position = rnd.randint(0, self.grid.shape[1] - 1)
            self.grid[self.grid.shape[0] - 1, rnd_south_col_position] = Marker.START
            self._start_location = (self.grid.shape[0] - 1, rnd_south_col_position)
            self.empty_Spaces.remove((self.grid.shape[0] - 1, rnd_south_col_position))

        elif rnd_start_location == MazeDirection.WEST:
            # Start position is the left most edge
            rnd_west_row_position = rnd.randint(0, self.grid.shape[0] - 1)
            self.grid[rnd_west_row_position, 0] = Marker.START
            self._start_location = (rnd_west_row_position, 0)
            self.empty_Spaces.remove((rnd_west_row_position, 0))

        elif rnd_start_location == MazeDirection.EAST:
            # Start position is the right most edge
            rnd_east_row_position = rnd.randint(0, self.grid.shape[0] - 1)
            self.grid[rnd_east_row_position, self.grid.shape[1] - 1] = Marker.START
            self._start_location = (rnd_east_row_position, self.grid.shape[1] - 1)
            self.empty_Spaces.remove((rnd_east_row_position, self.grid.shape[1] - 1))


a = Maze([5, 5])
print(a.grid)

