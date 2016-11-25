import random
import numpy as np
from buf import StateBuffer

class Agent:
	def __init__(self,
					environment,
					replay_memory,
					deep_q_network,
					history_length = 4,
					cols = 64,
					rows = 64,
					batch_size = 32,
					exploration_rate_start = 1,
					exploration_rate_end = 0.1,
					exploration_decay_steps = 10000,
					exploration_rate_test = 0.05,
					train_steps = 250000,
					start_epoch = 0,
					target_steps = 10000,
					train_frequency = 4,
					train_repeat = 1,
					random_starts = 10
				):
		self.env = environment
		self.mem = replay_memory
		self.net = deep_q_network
		self.buf = StateBuffer(history_length = history_length, cols = cols, rows = rows, batch_size = batch_size)
		self.num_actions = self.env.numActions()
		self.random_starts = random_starts
		self.history_length = history_length

		self.exploration_rate_start = exploration_rate_start
		self.exploration_rate_end = exploration_rate_end
		self.exploration_decay_steps = exploration_decay_steps
		self.exploration_rate_test = exploration_rate_test
		self.total_train_steps = start_epoch * train_steps

		self.target_steps = target_steps
		self.train_frequency = train_frequency
		self.train_repeat = train_repeat

	def _restartRandom(self):
		self.env.restart()
		for i in xrange(random.randint(self.history_length, self.random_starts) + 1):
			action, reward, board, terminal = self.step(1)
			self.buf.add(board)

	def _explorationRate(self):
		if self.total_train_steps < self.exploration_decay_steps:
			return self.exploration_rate_start - self.total_train_steps * (self.exploration_rate_start - self.exploration_rate_end) / self.exploration_decay_steps
		else:
			return self.exploration_rate_end

	def step(self, exploration_rate):
		if random.random() < exploration_rate:
			action = random.randrange(self.num_actions)
		else:
			state = self.buf.getStateMinibatch()
			qvalues = self.net.predict(state)
			assert len(qvalues[0]) == self.num_actions
			action = np.argmax(qvalues[0])

		reward = self.env.act(action)
		board = self.env.getBoard()
		terminal = self.env.isTerminal()

		self.buf.add(board)

		if terminal:
			self.env.restart()

		self.callback.on_step(action, reward, terminal, board, exploration_rate)

		return action, reward, board, terminal

	def train(self, train_steps, epoch = 0):
		for i in xrange(train_steps):
			action, reward, board, terminal = self.step(self._explorationRate())
			self.mem.add(action, reward, board, terminal)
			if self.target_steps and i % self.target_steps == 0:
				self.net.update_target_network()
			if self.mem.count > self.mem.batch_size and i % self.train_frequency == 0:
				for j in xrange(self.train_repeat):
					minibatch = self.mem.getMinibatch()
					self.net.train(minibatch, epoch)
			self.total_train_steps += 1

	def test(self, test_steps, epoch = 0):
		self._restartRandom()
		for i in xrange(test_steps):
			self.step(self.exploration_rate_test)

	def play(self, num_games):
		self._restartRandom()
		for i in xrange(num_games):
			terminal = False
			while not terminal:
				action, reward, screen, terminal = self.step(self.exploration_rate_test)
				self.mem.add(action, reward, screen, terminal)

