import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import time as tmr

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

    @staticmethod
    def get_direction(int_direction):
        if int_direction == MazeDirection.NORTH:
            return "north"
        elif int_direction == MazeDirection.SOUTH:
            return "south"
        elif int_direction == MazeDirection.WEST:
            return "west"
        elif int_direction == MazeDirection.EAST:
            return "east"
        else:
            raise ValueError("Invalid integer representation of a direction.")


class Marker:
    """
    These are the different elements on the grid board
    """
    EMPTY = 0
    GOAL = 9
    START = 2
    TRAP = 3
    TRIGGERED_TRAP = 4  # Signifies a state where a trap has been activated by agent move

    # The agent will not appear on the grid, but instead on the foreground layer (the agent layer)
    AGENT = 1


class RewardValue:
    """
    This keeps track of the rewards and associated values
    """
    REACHED_GOAL = 50.0  # Agent reaches the goal marker
    TRAP_HIT = -1.0  # Agent hits a trap
    OUTSIDE_MAZE = -0.5  # Agent attempted to make move that would take it out of bounds
    NO_REWARD = -0.1  # Agent gets -10 for each move made


class Maze:
    """
     The maze class is the game board. It will auto-generate a starting position and an ending goal,
     both of which must be on the board edges. Users of this class need to specify the dimensions of the
     board.
    """
    p_offset = 3  # this is the offset fort he edge borders
    trap_threshold = 0.3  # This represents the max percentage of traps that can be generated on a given maze board

    def __init__(self, tuple_size, int_obstacle_count=0):
        """
        :param tuple_size: [rows, cols] the size of the maze board
        :param int_obstacle_count: number of obstacles (traps)
        """

        # Keep track of the dimensions of the maze
        self.MazeDimension = tuple_size
        self.ObstacleCount = int_obstacle_count
        # The Game board, initialized to a numpy zeros array
        self.grid = np.zeros(tuple_size, dtype=int)  # This layer contains the start, end and all traps

        # Keep track of traps
        self._traps = []  # a list of coords [row, col] containing the trap

        # Private properties for the start location and end location
        self._start_location = ()  # Where the player starts. It is fixed for a given environment
        self._goal_location = ()  # Where the goal is. This is is fixed

        self.create_entry_exit()
        self._agent_position = self._start_location  # Tuple (row, col) keeps track of where agent is located
        # Set the agent position
        self._set_player(self._start_location)

        # Generate traps
        self.generate_traps(int_obstacle_count)

    @property
    def start_location(self):
        return self._start_location

    @property
    def goal_location(self):
        return self._goal_location

    @property
    def traps(self):
        return self._traps

    @property
    def agent_position(self):
        return self._agent_position

    def create_entry_exit(self):
        """
        places an entry mark and goal mark on the game board.
        :return: void
        """

        """
        First we determine the start position of the player. It must be on the edges (either north, south, west, east)
        We'll use a randomization to randomly pick which one it will be
        """
        rnd_start_location = rnd.randint(0, 3)  # get a random number between 0 and 3 inclusive
        if rnd_start_location == MazeDirection.NORTH:
            # the starting position needs to be on the top edge
            rnd_north_col_position = rnd.randint(0 + Maze.p_offset, self.grid.shape[1] - 1 - Maze.p_offset)
            self.grid[0, rnd_north_col_position] = Marker.START
            self._start_location = (0, rnd_north_col_position)

        elif rnd_start_location == MazeDirection.SOUTH:
            # Starting position is the outer bottom edge
            rnd_south_col_position = rnd.randint(0 + Maze.p_offset, self.grid.shape[1] - 1 - Maze.p_offset)
            self.grid[self.grid.shape[0] - 1, rnd_south_col_position] = Marker.START
            self._start_location = (self.grid.shape[0] - 1, rnd_south_col_position)

        elif rnd_start_location == MazeDirection.WEST:
            # Start position is the left most edge
            rnd_west_row_position = rnd.randint(0 + Maze.p_offset, self.grid.shape[0] - 1 - Maze.p_offset)
            self.grid[rnd_west_row_position, 0] = Marker.START
            self._start_location = (rnd_west_row_position, 0)

        elif rnd_start_location == MazeDirection.EAST:
            # Start position is the right most edge
            rnd_east_row_position = rnd.randint(0 + Maze.p_offset, self.grid.shape[0] - 1 - Maze.p_offset)
            self.grid[rnd_east_row_position, self.grid.shape[1] - 1] = Marker.START
            self._start_location = (rnd_east_row_position, self.grid.shape[1] - 1)
        else:
            raise RuntimeError("Something went wrong determining trap direction")

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
            rnd_col_pos = rnd.randint(0 + Maze.p_offset, self.grid.shape[1] - 1 - Maze.p_offset)
            return self.grid.shape[0] - 1, rnd_col_pos  # return a tuple
        elif self.start_location[0] == self.grid.shape[0] - 1:
            # -- SOUTH ---
            # Then a goal should be on the opposite end - NORTH
            rnd_col_pos = rnd.randint(0 + Maze.p_offset, self.grid.shape[1] - 1 - Maze.p_offset)
            return 0, rnd_col_pos

        # WEST
        if self.start_location[1] == 0:
            # Make a goal on the opposite side (EAST)
            # get a random row (up -down) position
            rnd_row_pos = rnd.randint(0 + Maze.p_offset, self.grid.shape[0] - 1 - Maze.p_offset)
            return rnd_row_pos, self.grid.shape[1] - 1
        elif self.start_location[1] == self.grid.shape[1] - 1:
            # EAST - return a goal position on the left (west) side
            rnd_row_pos = rnd.randint(0 + Maze.p_offset, self.grid.shape[0] - 1 - Maze.p_offset)
            return rnd_row_pos, 0

    def make_move(self, direction):
        """
        This is the only way to assess and control the agent's movement in the game environment
        Perform an action on the environment. Essentially allows the player to move in the maze
        one step at a time.
        :param direction: a direction to move (integer), one space up, down, left, right (NSWE) -- which is equivalent to our action spaces
        :return: observation, reward, done, info
        """
        # TODO: make_move method, add rewards and return observations etc
        """
        We will have to make sure that the direction chosen is a valid move.
        We also have to do collision checking (hitting traps, reaching the goal)
        We also have to ensure player does not go outside of maze
        """

        if direction == MazeDirection.NORTH:
            requested_move = (self._agent_position[0] - 1, self._agent_position[1])  # NORTH: move up one row
            # Validate
            if self.validate_agent_move(requested_move) == 0:
                #  Set the new player position, return a state / obs and reward
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.NO_REWARD, False, []
            elif self.validate_agent_move(requested_move) == -1:
                #  out of bounds, make no changes, return a negative reward and an obs
                return self.get_maze_state(), RewardValue.OUTSIDE_MAZE, False, []
            elif self.validate_agent_move(requested_move) == -2:
                # Agent has hit a trap, make sure to mark the trap as triggered, and send a negative reward
                # Update the player
                self._set_player(requested_move)
                self.grid[requested_move] = Marker.TRIGGERED_TRAP
                return self.get_maze_state(), RewardValue.TRAP_HIT, False, []
            elif self.validate_agent_move(requested_move) == 1:
                # Agent has reached the goal. Return positive reward, done = true, set agent to correct position
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.REACHED_GOAL, True, []
        elif direction == MazeDirection.SOUTH:
            requested_move = (self._agent_position[0] + 1, self._agent_position[1])  # SOUTH: move down one row
            if self.validate_agent_move(requested_move) == 0:
                #  Set the new player position, return a state / obs and reward
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.NO_REWARD, False, []
            elif self.validate_agent_move(requested_move) == -1:
                #  out of bounds, make no changes, return a negative reward and an obs
                return self.get_maze_state(), RewardValue.OUTSIDE_MAZE, False, []
            elif self.validate_agent_move(requested_move) == -2:
                # Agent has hit a trap, set the player to the starting position, return a negative reward
                self._set_player(requested_move)
                self.grid[requested_move] = Marker.TRIGGERED_TRAP
                return self.get_maze_state(), RewardValue.TRAP_HIT, False, []
            elif self.validate_agent_move(requested_move) == 1:
                # Agent has reached the goal. Return positive reward, done = true, set agent to correct position
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.REACHED_GOAL, True, []
        elif direction == MazeDirection.WEST:
            requested_move = (self._agent_position[0], self._agent_position[1] - 1)  # WEST: move left one col
            if self.validate_agent_move(requested_move) == 0:
                #  Set the new player position, return a state / obs and reward
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.NO_REWARD, False, []
            elif self.validate_agent_move(requested_move) == -1:
                #  out of bounds, make no changes, return a negative reward and an obs
                return self.get_maze_state(), RewardValue.OUTSIDE_MAZE, False, []
            elif self.validate_agent_move(requested_move) == -2:
                # Agent has hit a trap,  return a negative reward
                self._set_player(requested_move)
                self.grid[requested_move] = Marker.TRIGGERED_TRAP
                return self.get_maze_state(), RewardValue.TRAP_HIT, False, []
            elif self.validate_agent_move(requested_move) == 1:
                # Agent has reached the goal. Return positive reward, done = true, set agent to correct position
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.REACHED_GOAL, True, []
        elif direction == MazeDirection.EAST:
            requested_move = (self._agent_position[0], self._agent_position[1] + 1)  # EAST: move right one col
            if self.validate_agent_move(requested_move) == 0:
                #  Set the new player position, return a state / obs and reward
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.NO_REWARD, False, []
            elif self.validate_agent_move(requested_move) == -1:
                #  out of bounds, make no changes, return a negative reward and an obs
                return self.get_maze_state(), RewardValue.OUTSIDE_MAZE, False, []
            elif self.validate_agent_move(requested_move) == -2:
                # Agent has hit a trap, return a negative reward
                self._set_player(requested_move)
                self.grid[requested_move] = Marker.TRIGGERED_TRAP
                return self.get_maze_state(), RewardValue.TRAP_HIT, False, []
            elif self.validate_agent_move(requested_move) == 1:
                # Agent has reached the goal. Return positive reward, done = true, set agent to correct position
                self._set_player(requested_move)
                return self.get_maze_state(), RewardValue.REACHED_GOAL, True, []
        else:
            raise ValueError("Invalid Direction.")

    def validate_agent_move(self, desired_move):
        """
        :param desired_move: Tuple(row, col) of the move the player wants to make
        :return: Integer describing the results of the desired move:
        Validate move
        =============
        return value: 0 : no problem, unexceptional move
        return value: -1 : out of bounds
        return value: -2 : hit a trap
        return value: 1 : ** reached goal **

        """
        # Break down the row, col
        row = desired_move[0]
        col = desired_move[1]

        # Check for out of bounds (that is, agent making a move beyond the dimensions of the maze board
        if row < 0 or row > self.grid.shape[0] - 1:
            return -1

        if col < 0 or col > self.grid.shape[1] - 1:
            return -1

        # check for hitting a trap
        if desired_move in self._traps:
            return -2

        # check for reaching a goal
        if desired_move == self._goal_location:
            return 1

        return 0  # Validation has passed, return the default value of 0

    def generate_traps(self, trap_count=0):
        """
        This method will randomly place traps onto the maze board. The amount of traps depends on the size
        of the playing surface. There should be no more than 40% of the board covered in traps
        :param trap_count:
        :return:
        """

        # Firstly we must validate input to make sure there are no whacky trap count values supplied
        # For starters, we don't want more than threshold % of the grid to contain traps. This includes the grid size
        # minus two (which includes start and goal)

        actual_grid_size = self.grid.size - 2  # The grid size minus two (goal and start)
        max_traps = int(Maze.trap_threshold * actual_grid_size)  # The maximum traps size (40%)

        # This is going to be the total calculated trap count.
        # final_trap_count = 0

        # Check the input for trap_count. If it's zero or it's greater than max_traps, we'll
        # change it automatically to a random amount between max_traps and max_traps - deviation

        if trap_count == 0 or trap_count > max_traps:
            trap_count_deviation = rnd.randint(-5, 0)
            final_trap_count = max_traps + trap_count_deviation
        else:
            final_trap_count = trap_count

        trap_tally = 0  # This keeps track of how many traps were successfully placed in the maze

        while trap_tally <= final_trap_count:
            rnd_row = rnd.randint(0, self.grid.shape[0] - 1)
            rnd_col = rnd.randint(0, self.grid.shape[1] - 1)

            if self.trap_placement_is_ok((rnd_row, rnd_col)):
                trap_tally += 1  # increment. Otherwise, repeat the loop
                self.grid[rnd_row, rnd_col] = Marker.TRAP
                self._traps.append((rnd_row, rnd_col))  # Add the traps to the list

        # TODO: Once the loop has ended, we should do some kind of check at the board as a whole
        #  To make sure there aren't any weird things like straight lines across rows cols or diagonally

    def _set_player(self, placement):
        """
        Sets a player at placement(row, col)
        :param placement: a Tuple(row, col)
        :return: void
        """

        # Get the player's current position before moving it
        player_old_position = self._agent_position

        # This method should only be executed after all validation
        self.grid[placement] = Marker.AGENT
        self._agent_position = placement

        # Process the old position (check if it was a start, or a trap and restore the correct representation in the array)
        if player_old_position in self.traps:
            self.grid[player_old_position] = Marker.TRAP
        elif player_old_position == self._start_location:
            self.grid[player_old_position] = Marker.START
        else:
            self.grid[player_old_position] = Marker.EMPTY

        # print("New Position: ", self._agent_position)
        # print("Old position: ", player_old_position)

    def trap_placement_is_ok(self, placement):
        """
        Basic trap placement rules
        1) Trap can not be placed on a goal, or start position
        2) Trap cannot be in a safe (no-go) zone around a goal or start position

        :param placement: Tuple (row, col) of the desired placement of trap
        :return: bool - True if it's valid, else False.
        """

        # We get the safe zone for both the goal and start position
        no_go_zone_start_position = self._get_safe_zone(self._start_location)
        no_go_zone_goal_position = self._get_safe_zone(self._goal_location)

        if placement in no_go_zone_start_position:
            return False

        if placement in no_go_zone_goal_position:
            return False

        return True

    def _get_safe_zone(self, row_col):
        """
        :param row_col: a tuple containing [row, col] of the safe zone you want to have
        :return: a list[] of tuples(row, col) of the no-go safe zone around row_col
        This usually is for trap placement to ensure that no traps are placed in any coord in the safe zone
        Check out the github for an example of what a 'safe zone' looks like
        """

        #  Separate out the row, col to make it easier to read
        c_row = row_col[0]
        c_col = row_col[1]

        #  NORTH
        if c_row == 0:
            # Create the no-go-zone around the spot in question
            output = [(c_row, c_col), (c_row, c_col - 1), (c_row, c_col - 2), (c_row + 1, c_col - 1), (c_row + 1, c_col - 2),
                      (c_row + 1, c_col), (c_row + 2, c_col), (c_row, c_col + 1), (c_row, c_col + 2), (c_row + 1, c_col + 1),
                      (c_row + 1, c_col + 2)]

            return output
        elif c_row == self.grid.shape[0] - 1:
            #  SOUTH
            output = [(c_row, c_col), (c_row, c_col - 1), (c_row, c_col - 2), (c_row - 1, c_col - 1), (c_row - 1, c_col - 2),
                      (c_row - 1, c_col), (c_row - 2, c_col), (c_row, c_col + 1), (c_row, c_col + 2), (c_row - 1, c_col + 1),
                      (c_row - 1, c_col + 2)]
            return output

        #  WEST
        if c_col == 0:
            output = [(c_row, c_col), (c_row - 1, c_col), (c_row - 2, c_col), (c_row - 1, c_col + 1), (c_row - 2, c_col + 1),
                      (c_row, c_col + 1), (c_row, c_col + 2), (c_row + 1, c_col), (c_row + 2, c_col), (c_row + 1, c_col + 1),
                      (c_row + 2, c_col + 1)]
            return output
        elif c_col == self.grid.shape[1] - 1:
            # -- EAST
            output = [(c_row, c_col), (c_row - 1, c_col), (c_row - 2, c_col), (c_row - 1, c_col - 1), (c_row - 2, c_col - 1),
                      (c_row, c_col - 1), (c_row, c_col - 2), (c_row + 1, c_col), (c_row + 2, c_col), (c_row + 1, c_col - 1),
                      (c_row + 2, c_col - 1)]
            return output

    def print(self):
        """
        Prints the ndarray
        :return: void
        """
        print(self.grid, end="\r")
        tmr.sleep(0.5)

    def get_maze_state(self):
        """
        A short function for getting a the state of the game grid
        :return: numpy.ndarray
        """
        return self.grid

    def reset(self):
        """
        Reset: place the agent back at the starting position. Clear board and re-generate maze
        :return: initial observation
        """
        self.grid = np.zeros(self.MazeDimension, dtype=int)  # This layer contains the start, end and all traps

        # Keep track of traps
        self._traps = []  # a list of coords [row, col] containing the trap

        self.create_entry_exit()
        self._agent_position = self._start_location  # Tuple (row, col) keeps track of where agent is located
        # Set the agent position
        self._set_player(self._start_location)

        # Generate traps
        self.generate_traps(self.ObstacleCount)

        return self.get_maze_state()


# Testing
def testing():
    maze_dimension = (10, 10)
    env = Maze(maze_dimension)

    num_episodes = 15000
    reward_storage = []
    move_storage = []
    max_num_moves = 100  # a game (episode) cannot take more moves than this, otherwise it's too long.
                         # I calculated this number because it's slightly higher than the mean amount of moves

    num_valid_games_played = 0
    for i in range(num_episodes):
        done = None
        r_tally = 0
        num_moves = 0
        saved_states = []

        obs = env.reset()

        while not done:
            action = rnd.randint(0, 3)  # Represents our action space
            obs, reward, done, info = env.make_move(action)
            # print(obs)
            # tmr.sleep(.7)
            r_tally += reward
            num_moves += 1
            if num_moves >= max_num_moves:
                break  # Abort because the agent is taking too many moves

            saved_states.append(obs)
            if done:
                # print("Episode completed. Total Reward is: ", r_tally, "|| number of moves: ", num_moves)
                reward_storage.append(r_tally)
                move_storage.append(num_moves)
                num_valid_games_played += 1



