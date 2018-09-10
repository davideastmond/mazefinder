import random as rnd
import numpy as np

"""
This is a Q learning re-enforcement learning project. The object of this game is to navigate 
the maze to the goal, without hitting any traps.
"""
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


class RewardValue:
    """
    This keeps track of the rewards and associated values
    """
    REACHED_GOAL = 2000
    TRAP_HIT = -1000
    OUTSIDE_MAZE = -500


class Maze:
    """
     The maze class is hte game board. It will auto-generate a starting position and an ending goal,
     both of which must be on the board edges. Users of this class need to specify the dimensions of the
     board.
    """
    def __init__(self, tuple_size=(10, 10), int_obstacle_count=0):
        """
        :param tuple_size: [rows, cols] the size of the maze board
        :param int_obstacleCount: number of obstacles (traps)
        """
        # The Game board, initialized to a numpy zeros array
        self.grid = np.zeros([tuple_size[0], tuple_size[1]], dtype=int)

        # Private properties for the start location and end location
        self._start_location = ()  # Where the player starts. It is fixed for a given environment
        self._goal_location = ()  # Where the goal is. This is is fixed
        self.agent_position = ()  # Tuple (row, col) keeps track of where agent is located
        self.agent_player = None  # Keeps track of the agent object

        self.create_entry_exit()

        # Set the agent position
        self.agent_position = self._start_location

        # Generate traps
        self.generate_traps()

    @property
    def start_location(self):
        return self._start_location

    @property
    def goal_location(self):
        return self._goal_location

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

        elif rnd_start_location == MazeDirection.SOUTH:
            # Starting position is the outer bottom edge
            rnd_south_col_position = rnd.randint(0, self.grid.shape[1] - 1)
            self.grid[self.grid.shape[0] - 1, rnd_south_col_position] = Marker.START
            self._start_location = (self.grid.shape[0] - 1, rnd_south_col_position)

        elif rnd_start_location == MazeDirection.WEST:
            # Start position is the left most edge
            rnd_west_row_position = rnd.randint(0, self.grid.shape[0] - 1)
            self.grid[rnd_west_row_position, 0] = Marker.START
            self._start_location = (rnd_west_row_position, 0)

        elif rnd_start_location == MazeDirection.EAST:
            # Start position is the right most edge
            rnd_east_row_position = rnd.randint(0, self.grid.shape[0] - 1)
            self.grid[rnd_east_row_position, self.grid.shape[1] - 1] = Marker.START
            self._start_location = (rnd_east_row_position, self.grid.shape[1] - 1)

        # Finally we assign the self.goal_position property and place it on the grid
        self._goal_location = self._determine_exit()
        self.grid[self._goal_location] = Marker.GOAL

    def _determine_exit(self):
        """
        using the starting location property, determine where a goal will be
        :return: Tuple (int, int) indicating the location of the exit
        """

        # NORTH
        if self.start_location[0] == 0:
            # The starting location is in the top row, therefore, goal should be south
            # We need to get a random column pos
            rnd_col_pos = rnd.randint(0, self.grid.shape[1] - 1)
            return self.grid.shape[0] - 1, rnd_col_pos  # return a tuple
        elif self.start_location[0] == self.grid.shape[0] - 1:
            # -- SOUTH ---
            # Then a goal should be on the opposite end - NORTH
            rnd_col_pos = rnd.randint(0, self.grid.shape[1] - 1)
            return 0, rnd_col_pos

        # WEST
        if self.start_location[1] == 0:
            # Make a goal on the opposite side (EAST)
            # get a random row (up -down) position
            rnd_row_pos = rnd.randint(0, self.grid.shape[0] - 1)
            return rnd_row_pos, self.grid.shape[1] - 1
        elif self.start_location[1] == self.grid.shape[1] - 1:
            # EAST - return a goal position on the left (west) side
            rnd_row_pos = rnd.randint(0, self.grid.shape[0] - 1)
            return rnd_row_pos, 0

    def make_move(self, direction):
        """
        Perform an action on the environment. Essentially allows the player to move in the maze
        one step at a time.
        :param direction: a direction to move, one space up, down, left, right
        :return: observation, reward, done, info
        """
        # TODO: make_move method
        """
        We will have to make sure that the direction chosen is a valid move.
        We also have to do collision checking (hitting traps, reaching the goal)
        We also have to ensure player does not go outside of maze
        """

    def generate_traps(self, trap_count=0):
        """
        This method will randomly place traps onto the maze board. The amount of traps depends on the size
        of the playing surface. There should be no more than 40% of the board covered in traps
        :param trap_count:
        :return:
        """

        # Firstly we must valid input to make sure there are no whacky trap count values supplied
        # For starters, we don't want more than 40% of the grid to contain traps. This includes the grid size
        # minus two (which includes start and goal)

        actual_grid_size = self.grid.size - 2 # The grid size minus two (goal and start)
        max_traps = int(0.4 * actual_grid_size)  # The maximum traps size (40%)


        # This is going to be the total calculated trap count.
        final_trap_count = 0

        # Check the input for trap_count. If it's zero or it's greater than max_traps, we'll
        # change it automatically to a random amount between max_traps and max_traps - deviation

        if trap_count == 0 or trap_count > max_traps:
            trap_count_deviation = rnd.randint(-5, 0)
            final_trap_count = max_traps + trap_count_deviation
        else:
            final_trap_count = trap_count

        trap_tally = 0
        while trap_tally <= final_trap_count:
            rnd_row = rnd.randint(0, self.grid.shape[0] - 1)
            rnd_col = rnd.randint(0, self.grid.shape[1] - 1)

            placement_ok = self.trap_placement_is_ok((rnd_row, rnd_col))
            if placement_ok:
                trap_tally += 1  # increment. Otherwise, repeat the loop

        #  Once the loop has ended, we should do some kind of check at the board as a whole
        #  To make sure there aren't any weird things like straight lines across rows cols or diagonally

        # TODO: generate_trap method.
        # TODO: create a separate trap_placement_validate method that determines

    def trap_placement_is_ok(self, placement):
        """
        Basic trap placement rules
        1) Trap can not be placed on a goal, or start position
        2) Trap cannot surround (block) a goal or start position

        :param placement: Tuple (row, col) of the desired placement of trap
        :return: bool - True if it's valid, else False.
        """
        return False
# Testing
a = Maze([5, 5])
print(a.grid.size)

