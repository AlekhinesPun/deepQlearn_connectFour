import numpy as np
import random

class ReplayMemory:
	def __init__(self, size=1000000, cols=64, rows=64, history_length=4, batch_size=32):
		self.size = size
		self.actions = np.empty(self.size, dtype = np.uint8)
		self.rewards = np.empty(self.size, dtype = np.integer)
		self.boards = np.empty((self.size, cols, rows), dtype = np.uint8)
		self.terminals = np.empty(self.size, dtype = np.bool)
		self.history_length = history_length
		self.dims = (cols, rows)
		self.batch_size = batch_size
		self.count = 0
		self.current = 0

		self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)
		self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.uint8)

	def add(self, action, reward, board, terminal):
		assert board.shape == self.dims
		# NB! board is post-state, after action and reward
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.boards[self.current, ...] = board
		self.terminals[self.current] = terminal
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.size

	def getState(self, index):
		assert self.count > 0, "replay memory is empty, use at least one random step"
		index = index % self.count
		if index >= self.history_length - 1:
			return self.boards[(index - (self.history_length - 1)):(index + 1), ...]
		else:
			indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
			return self.boards[indexes, ...]

	def getMinibatch(self):
		assert self.count > self.history_length
		indexes = []
		while len(indexes) < self.batch_size:
			while True:
				index = random.randint(self.history_length, self.count - 1)
				if index >= self.current and index - self.history_length < self.current:
					continue
				if self.terminals[(index - self.history_length):index].any():
					continue
				break

			self.prestates[len(indexes), ...] = self.getState(index - 1)
			self.poststates[len(indexes), ...] = self.getState(index)
			indexes.append(index)

		actions = self.actions[indexes]
		rewards = self.rewards[indexes]
		terminals = self.terminals[indexes]
		return self.prestates, actions, rewards, self.poststates, terminals