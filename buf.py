import numpy as np

class StateBuffer:
	def __init__(self,
					history_length = 4,
					cols = 64,
					rows = 64,
					batch_size = 32
				):
		self.history_length = history_length
		self.dims = (cols, rows)
		self.batch_size = batch_size
		self.buffer = np.zeros((self.batch_size, self.history_length) + self.dims, dtype=np.uint8)

	def add(self, observation):
		assert observation.shape == self.dims
		self.buffer[0, :-1] = self.buffer[0, 1:]
		self.buffer[0, -1] = observation

	def getState(self):
		return self.buffer[0]

	def getStateMinibatch(self):
		return self.buffer

	def reset(self):
		self.buffer *= 0
