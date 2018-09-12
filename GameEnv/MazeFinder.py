"""
This is the Maze Finder Environment for Q RL learning
"""
from GameEnv.env_classes.Maze import Maze


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
        self.Game.reset(self.Dimensions)  # Reset the maze board to the default, initial dimensions

    def render(self):
        self.Game.print()  # Essentially prints the game board
