"""
This is the Maze Finder Environment for Q RL learning
"""
from .env_classes.Maze import Maze, MazeDirection


class MazeFinder:

    def __init__(self, tuple_size):
        """
        Create a maze environment
        :param tuple_size: A tuple indicating the dimensions of the maze board
        """
        self.Game = Maze(tuple_size)
        self.Dimensions = tuple_size  # keep track of the dimensions of the maze

    def step(self, step_action):
        """
        step_action: an int indicating a direction to move (as defined by Maze.MazeDirection)
        This is where the agent makes a move on the environment, and we get back some data as specified below in the return
        :return: observation, reward, done, info
        """
        return self.Game.make_move(step_action)

    def reset(self):
        """
        Reset the maze board to the default, initial dimensions
        :return: void
        """
        self.Game.reset()

    def render(self):
        """
        Raw output of maze board contents
        :return:
        """
        self.Game.print()

    @staticmethod
    def get_action_space():
        """
        Static method. Returns a list of possible actions an agent could take
        :return: list of 4 possible actions (up down, left right)
        """
        return [MazeDirection.NORTH, MazeDirection.SOUTH, MazeDirection.WEST, MazeDirection.EAST]  # return a list of valid actions
