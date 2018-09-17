# To upload:
# from google.colab import files
# uploaded = files.upload()


# The following code will be used to download the models we train
# from google.colab import files
# files.download('Path of file to be downloaded')

import numpy as np
import h5py
from GameEnv.MazeFinder import MazeFinder

NUM_EPISODES = 10000
GAMMA = 0.99
STATE_SHAPE = [10, 10]
COUNT_STATES = 0
AGENT_NUM = 0


def count_up():
    global COUNT_STATES
    COUNT_STATES += 1


class Agent:
    def __init__(self, num_actions, training, load_file=None, agent_path=None):

        self.training = training
        self.num_actions = num_actions

        self.epsilon_greedy = EpsilonGreedy(start_value=1.0,
                                            end_value=0.1,
                                            num_iterations=1e4,
                                            num_actions=self.num_actions,
                                            epsilon_testing=0.01)

        if self.training:

            self.replay_memory = ReplayMemory(size=200000,
                                              num_actions=self.num_actions)
        else:

            self.replay_memory = None

        self.network = NeuralNetwork(self.num_actions, self.replay_memory)

        if load_file is not None:
            self.network.load(file_path=agent_path, name=load_file)

        self.episode_rewards = []

    def reset_episode_rewards(self):
        """Reset the log of episode-rewards."""
        self.episode_rewards = []

    def get_move(self, _state):

        n_q_values = self.network.get_q_values(p_state=_state)[0]

        n_action = self.epsilon_greedy.get_action(q_values=q_values, iteration=COUNT_STATES)

        return n_q_values, n_action

    def remember_move(self, p_state, p_q_values, p_action, p_reward, p_end_episode):
        # Saves the state, values actions and rewards into replay memory
        self.replay_memory.add(state=p_state,
                               q_values=p_q_values,
                               action=p_action,
                               reward=p_reward,
                               end_episode=p_end_episode)

    def freeze_agent(self):
        self.replay_memory = None
        self.training = False
        self.epsilon_greedy.set_training(False)

    def copy_agent(self, p_agent):
        self.network.model.set_weights(p_agent.network.model.get_weights())


class ReplayMemory:

    def __init__(self, size, num_actions, discount_factor=GAMMA):
        """
        :param size:
            Capacity of the replay-memory. This is the number of states.
        :param num_actions:
            Number of possible actions in the game-environment.
        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        self.states = np.zeros(shape=[size] + STATE_SHAPE)

        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        self.actions = np.zeros(shape=size, dtype=np.int)
        self.rewards = np.zeros(shape=size, dtype=np.float)
        self.end_life = np.zeros(shape=size, dtype=np.bool)
        self.size = size

        self.estimation_errors = np.zeros(shape=size, dtype=np.float)
        self.discount_factor = discount_factor
        self.error_threshold = 0.1

        self.num_used = 0
        self.idx = []

    def is_full(self):
        """Return boolean whether the replay-memory is full."""
        return self.num_used == self.size

    def used_fraction(self):
        """Return the fraction of the replay-memory that is used."""
        return self.num_used / self.size

    def reset(self):
        """Reset the replay-memory so it is empty."""
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.
        :param q_values:
            The estimated Q-values for the state.
        :param action:
            The action taken by the agent in this state of the game.
        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.
        :param end_life:
            Boolean whether the agent has lost a life in this state.

        :param end_episode:
            Boolean whether the agent has lost all lives aka. game over
            aka. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elements in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life

            # Note that the reward is limited. This is done to stabilize
            # the training of the Neural Network.
            # self.rewards[k] = np.clip(reward, -1.0, 1.0)
            self.rewards[k] = reward

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we now know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statistics later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This loop could be implemented entirely in NumPy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used - 1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value with the better estimate.
            self.q_values[k, action] = action_value

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of states
        and Q-values for use in training of the Neural Network.
        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural Network because
        they have high estimation errors.

        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural Network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err < self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estimation errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

        print("Low_samples {}, Highsamples {}, memory use {}".format(len(self.idx_err_lo),
                                                                     len(self.idx_err_lo),
                                                                     self.num_used))

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.
        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.

        if self.num_samples_err_lo > 0:
            idx_lo = np.random.choice(self.idx_err_lo,
                                      size=self.num_samples_err_lo,
                                      replace=False)
        else:
            idx_lo = [0]
        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def get_random_batch(self, batch_size):
        indices = [x for x in range(self.num_used)]
        indices = np.random.choice(indices, size=batch_size)

        states_batch = self.states[indices]
        q_values_batch = self.q_values[indices]

        return states_batch, q_values_batch


class LinearControlSignal:
    """
    A control signal that changes linearly over time.
    This is used to change e.g. the learning-rate for the optimizer
    of the Neural Network, as well as other parameters.

    TensorFlow has functionality for doing this, but it uses the
    global_step counter inside the TensorFlow graph, while we
    want the control signals to use a state-counter for the
    game-environment. So it is easier to make this in Python.
    """

    def __init__(self, start_value, end_value, num_iterations, repeat=False):
        """
        Create a new object.
        :param start_value:
            Start-value for the control signal.
        :param end_value:
            End-value for the control signal.
        :param num_iterations:
            Number of iterations it takes to reach the end_value
            from the start_value.
        :param repeat:
            Boolean whether to reset the control signal back to the start_value
            after the end_value has been reached.
        """

        # Store arguments in this object.
        self.start_value = start_value
        self.end_value = end_value
        self.num_iterations = num_iterations
        self.repeat = repeat

        # Calculate the linear coefficient.
        self._coefficient = (end_value - start_value) / num_iterations

    def get_value(self, iteration):
        """Get the value of the control signal for the given iteration."""

        if self.repeat:
            iteration %= self.num_iterations

        if iteration < self.num_iterations:
            value = iteration * self._coefficient + self.start_value
        else:
            value = self.end_value

        return value


class EpsilonGreedy:
    """
    The epsilon-greedy policy either takes a random action with
    probability epsilon, or it takes the action for the highest
    Q-value.

    If epsilon is 1.0 then the actions are always random.
    If epsilon is 0.0 then the actions are always argmax for the Q-values.
    Epsilon is typically decreased linearly from 1.0 to 0.1
    and this is also implemented in this class.
    During testing, epsilon is usually chosen lower, e.g. 0.05 or 0.01
    """

    def __init__(self, num_actions,
                 epsilon_testing=0.05,
                 num_iterations=1e5,
                 start_value=1.0,
                 end_value=0.1,
                 training=True,
                 repeat=False):
        """

        :param num_actions:
            Number of possible actions in the game-environment.
        :param epsilon_testing:
            Epsilon-value when testing.
        :param num_iterations:
            Number of training iterations required to linearly
            decrease epsilon from start_value to end_value.

        :param start_value:
            Starting value for linearly decreasing epsilon.
        :param end_value:
            Ending value for linearly decreasing epsilon.
        :param repeat:
            Boolean whether to repeat and restart the linear decrease
            when the end_value is reached, or only do it once and then
            output the end_value forever after.
        """

        # Store parameters.
        self.num_actions = num_actions
        self.epsilon_testing = epsilon_testing
        self.training = training

        # Create a control signal for linearly decreasing epsilon.
        self.epsilon_linear = LinearControlSignal(num_iterations=num_iterations,
                                                  start_value=start_value,
                                                  end_value=end_value,
                                                  repeat=repeat)

    def get_epsilon(self, iteration):
        """
        Return the epsilon for the given iteration.
        If training==True then epsilon is linearly decreased,
        otherwise epsilon is a fixed number.
        """

        if self.training:
            epsilon = self.epsilon_linear.get_value(iteration=iteration)
        else:
            epsilon = self.epsilon_testing

        return epsilon

    def get_action(self, q_values, iteration):
        """
        Use the epsilon-greedy policy to select an action.

        :param q_values:
            These are the Q-values that are estimated by the Neural Network
            for the current state of the game-environment.

        :param iteration:
            This is an iteration counter. Here we use the number of states
            that has been processed in the game-environment.
        :param training:
            Boolean whether we are training or testing the
            Reinforcement Learning agent.
        :return:
            action (integer), epsilon (float)
        """

        epsilon = self.get_epsilon(iteration=iteration)

        # With probability epsilon.
        if np.random.random() < epsilon:
            # Select a random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            # Otherwise select the action that has the highest Q-value.
            action = np.argmax(q_values)

        return action

    def set_training(self, training_bool):
        """
        Sets the self.training bool for different epsilons
        """
        if isinstance(training_bool, bool):
            self.training = training_bool
        else:
            print("Input must be a boolean")


from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten
from keras import losses
from keras.models import load_model
from keras.initializers import RandomNormal


class NeuralNetwork:
    """
    Creates a neural network for Q-learning
    """

    def __init__(self, num_actions, replay_memory):

        self.replay_memory = replay_memory
        self.learning_rate = None

        init = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        self.model = Sequential()

        #     self.model.add(Dense(1024,activation='relu',input_dim=100,
        #                         kernel_initializer=init))

        self.model.add(Conv2D(16, (3, 3),
                              activation='relu',
                              input_shape=(10, 10, 1),
                              kernel_initializer=init))
        self.model.add(Conv2D(32, (3, 3),
                              activation='relu',
                              kernel_initializer=init))
        self.model.add(Conv2D(64, (3, 3),
                              activation='relu',
                              kernel_initializer=init))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu',
                             kernel_initializer=init))
        self.model.add(Dense(512, activation='relu',
                             kernel_initializer=init))
        self.model.add(Dense(512, activation='relu',
                             kernel_initializer=init))
        self.model.add(Dense(512, activation='relu',
                             kernel_initializer=init))

        self.model.add(Dense(num_actions, activation='linear'))

        self.model.compile(optimizer='rmsprop',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def save_model(self, file_path=None, name='agent'):
        """
        Save the current model
        """
        if file_path is None:
            self.model.save(name + '.h5')
        else:
            self.model.save(file_path + name + '.h5')

    def load(self, file_path=None, name='agent'):

        if file_path is None:
            self.model = load_model(name + '.h5')
        else:

            self.model = load_model(file_path + name + '.h5')

    def get_q_values(self, p_state):

        return self.model.predict_on_batch(p_state.reshape(1, 10, 10, 1))

    def optimize(self, batch_size, loss_limit, learning_rate, max_epochs=10):
        # self.replay_memory.prepare_sampling_prob(batch_size=batch_size)

        iterations_per_epoch = self.replay_memory.num_used / batch_size
        min_iterations = int(iterations_per_epoch)
        max_iterations = int(iterations_per_epoch * max_epochs)
        loss_memory = []

        for i in range(max_iterations):

            state_batch, q_values_batch = self.replay_memory.get_random_batch(batch_size=batch_size)

            loss = self.model.train_on_batch(state_batch.reshape(batch_size, 10, 10, 1),
                                             q_values_batch)

            loss_memory.append(loss)
            if i > 100:
                loss_mean = np.mean(loss_memory[-100:])
            else:
                loss_mean = np.mean(loss_memory[-i:])

            if i > min_iterations and loss_mean < loss_limit:
                return loss_memory
        return loss_memory


import random as rd
import matplotlib.pyplot as plt
# from google.colab import files


class Trainer:
    """
    This class is used to train Agents. The reasoning behind seperating the
    Trainer and Agent is because the Trainer will use 2 Agents (Training vs
    Frozen) and this allows for easier control
    """

    def __init__(self, env_instance, training_agent):
        """
        :param training_agent:
         The agent to be trained for given env
        """

        self.env = env_instance
        self.training_agent = training_agent

        self.learning_rate_control = LinearControlSignal(start_value=1e-3,
                                                         end_value=1e-5,
                                                         num_iterations=5e6)

        self.loss_limit_control = LinearControlSignal(start_value=0.1,
                                                      end_value=0.015,
                                                      num_iterations=5e6)

        self.max_epochs_control = LinearControlSignal(start_value=5.0,
                                                      end_value=10.0,
                                                      num_iterations=5e6)

        self.replay_fraction = LinearControlSignal(start_value=0.1,
                                                   end_value=1.0,
                                                   num_iterations=5e6)

    def train_one_episode(self):
        """
        Plays through one game with training agent as player one, return True if
        agent wins, False otherwise
        """

        t_state = self.env.reset()
        t_total_reward = 0
        t_done = False
        t_max_steps = 0
        while not self.training_agent.replay_memory.is_full() and not t_done and t_max_steps < 100:
            local_q_values, local_action = self.training_agent.get_move(t_state)
            nstate, nreward, ndone, _ = self.env.step(action)
            count_up()
            self.training_agent.replay_memory.add(state=nstate,
                                                  q_values=local_q_values,
                                                  action=local_action,
                                                  reward=nreward,
                                                  end_life=ndone)

            t_total_reward += reward
            t_max_steps += 1

        return total_reward

    def train_agent(self, num_games=None):

        if num_games is None:
            num_games = float('inf')

        num_games_done = 0

        self.game_history = []
        self.avg_game_history = []
        self.q_values_history = []
        self.reward_history = []
        self.avg_reward_history = []
        self.loss_history = []
        global AGENT_NUM
        optimized = False

        use_fraction = self.replay_fraction.get_value(iteration=COUNT_STATES)

        while num_games_done <= num_games:

            # How much of the replay-memory should be used.

            if optimized:
                use_fraction = self.replay_fraction.get_value(iteration=COUNT_STATES)
                optimized = False

            result = self.train_one_episode()

            self.reward_history.append(result)

            if result > 0:
                self.game_history.append(1)
            else:
                self.game_history.append(0)

            avg_winrate = np.mean(self.game_history[-100:])

            if num_games_done > 50:
                self.avg_reward_history.append(np.mean(self.reward_history[-50:]))
                self.avg_game_history.append(avg_winrate)

                if num_games_done % 1000 == 0:
                    print("Game Num: {} Avg Reward:{} Avg Winrate{}".format(num_games_done,
                                                                            self.avg_reward_history[-1],
                                                                            avg_winrate))

                    self.plot_statistics()

            if self.training_agent.replay_memory.is_full() \
                    or self.training_agent.replay_memory.used_fraction() > use_fraction:
                self.training_agent.replay_memory.update_all_q_values()
                self.q_values_history.append(np.mean(self.training_agent.replay_memory.q_values))

                learning_rate = self.learning_rate_control.get_value(iteration=COUNT_STATES)
                loss_limit = self.loss_limit_control.get_value(iteration=COUNT_STATES)
                max_epochs = self.max_epochs_control.get_value(iteration=COUNT_STATES)

                # Perform an optimization run on the Neural Network so as to
                # improve the estimates for the Q-values.
                # This will sample random batches from the replay-memory.
                loss_list = self.training_agent.network.optimize(learning_rate=learning_rate,
                                                                 batch_size=128,
                                                                 loss_limit=loss_limit,
                                                                 max_epochs=max_epochs)

                self.loss_history += loss_list
                self.training_agent.replay_memory.reset()

            num_games_done += 1

    def train_with_self_play(self):
        pass

    def plot_statistics(self):

        # plt.plot(self.game_history,label='wins')
        if len(self.game_history) > 100:
            plt.plot(self.avg_game_history, label='Avg winrate for last 100 games)')
            plt.ylabel("Win")
            plt.xlabel("No. of games played")
            plt.title("Games won by agents & Avg winrate of agent over last 100 games")
            plt.legend()
            plt.show()

        plt.plot(self.q_values_history, label='q_values Mean', marker='o')
        plt.ylabel("q_values")
        plt.xlabel("Optimizations")
        plt.title("Q-values mean")
        plt.legend()
        plt.show()

        plt.plot(self.reward_history, label="reward")
        plt.ylabel("Games")
        plt.xlabel("Reward")
        plt.title("Reward graph")
        plt.legend()
        plt.show()

        plt.plot(self.avg_reward_history, label="avg reward over last 30 games")
        plt.ylabel("Games")
        plt.xlabel("Reward")
        plt.title("Average reward graph")
        plt.legend()
        plt.show()

        if len(loss_history_array) > 0:
            loss_history_array = np.array(self.loss_history)
            loss = loss_history_array[:, 0]

            plt.plot(loss, label="loss")
            plt.xlabel("Optimizations")
            plt.title("Loss")
            plt.legend()
            plt.show()

            acc = 100 * loss_history_array[:, 1]
            plt.plot(acc, label="accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Optimizations")
            plt.title("Accuracy")
            plt.legend()
            plt.show()


env = MazeFinder((10, 10))
agent = Agent(num_actions=4, training=True)
trainer = Trainer(env, agent)
trainer.train_agent(num_games=13000)

agent.training = False

rewards = []

for x in range(100):
    state = env.reset()
    total_reward = 0
    done = False
    max_steps = 0
    while not done and max_steps < 100:
        q_values, action = agent.get_move(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        max_steps += 1

    rewards.append(total_reward)

plt.plot(rewards, label="rewards")
plt.ylabel("Rewards")
plt.xlabel("Games")
plt.title("Reward graph")
plt.legend()
plt.show()

trainer.train_agent(num_games=15000)

agent.training = False
rewards = []

for x in range(50):
    state = env.reset()
    total_reward = 0
    done = False
    max_steps = 0
    while not done and max_steps < 100:
        q_values, action = agent.get_move(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        max_steps += 1

    rewards.append(total_reward)

plt.plot(rewards, label="rewards", marker='o')
plt.ylabel("Rewards")
plt.xlabel("Games")
plt.title("Reward graph")
plt.legend()
plt.show()


# # The following code will be used to download the models we train
# from google.colab import files

# for x in range(29):
files.download('agent{}.h5'.format(AGENT_NUM - 1))


