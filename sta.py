import time
import sys
import numpy as np

class Statistics:
	def __init__(self, agent, net, mem, env):
		self.agt = agent
		self.net = net
		self.mem = mem
		self.env = env

		self.agt.callback = self
		self.net.callback = self

		self.start_time = time.clock()

	def reset(self):
		self.epoch_start_time = time.clock()
		self.num_steps = 0
		self.num_games = 0
		self.game_rewards = 0
		self.average_reward = 0
		self.min_game_reward = sys.maxint
		self.max_game_reward = -sys.maxint - 1
		self.last_exploration_rate = 1
		self.average_cost = 0

	def on_step(self, action, reward, terminal, board, exploration_rate):
		self.game_rewards += reward
		self.num_steps += 1
		self.last_exploration_rate = exploration_rate

		if terminal:
			self.num_games += 1
			self.average_reward += float(self.game_rewards - self.average_reward) / self.num_games
			self.min_game_reward = min(self.min_game_reward, self.game_rewards)
			self.max_game_reward = max(self.max_game_reward, self.game_rewards)
			self.game_rewards = 0

	def on_train(self, cost):
		self.average_cost += (cost - self.average_cost) / self.net.train_iterations

	def out(self, phase, epoch):
		current_time = time.clock()
		epoch_time = current_time - self.epoch_start_time
		steps_per_second = self.num_steps / epoch_time

		print "  num_games: %d, average_reward: %f, min_game_reward: %d, max_game_reward: %d" % (self.num_games, self.average_reward, self.min_game_reward, self.max_game_reward)
		print "  last_exploration_rate: %f, epoch_time, %ds, steps_per_second: %d" % (self.last_exploration_rate, epoch_time, steps_per_second)