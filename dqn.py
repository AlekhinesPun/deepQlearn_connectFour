from neon.backends import gen_backend
from neon.initializers import Xavier
from neon.layers import Affine, Conv
from neon.transforms import Rectlin
from neon.models import Model
from neon.layers import GeneralizedCost
from neon.transforms import SumSquared
from neon.optimizers import RMSProp
import numpy as np


class DeepQNetwork:

	def __init__(self, num_actions,
					batch_size = 32,
					discount_rate = 0.99,
					history_length = 4,
					cols = 64,
					rows = 64,
					clip_error = 1,
					min_reward = -1,
					max_reward = 1,
					batch_norm = False
				):
		self.num_actions = num_actions
		self.batch_size = batch_size
		self.discount_rate = discount_rate
		self.history_length = history_length
		self.board_dim = (cols, rows)
		self.clip_error = clip_error
		self.min_reward = min_reward
		self.max_reward = max_reward
		self.batch_norm = batch_norm

		self.be = gen_backend(backend='gpu',
							batch_size = self.batch_size,
							datatype= np.dtype('float32').type
							)

		self.input_shape = (self.history_length,) + self.board_dim + (self.batch_size,)
		self.input = self.be.empty(self.input_shape)
		self.input.lshape = self.input_shape # hack from simple_dqn "needed for convolutional networks"
		self.targets = self.be.empty((self.num_actions, self.batch_size))

		layers = self._createLayers(self.num_actions)
		self.model = Model(layers = layers)
		self.cost = GeneralizedCost(costfunc = SumSquared())
		# for l in self.model.layers.layers:
		# 	l.parallelism = 'Disabled'
		self.model.initialize(self.input_shape[:-1], cost = self.cost)
		self.optimizer = RMSProp(
							learning_rate = 0.002, 
							decay_rate = 0.95,
							stochastic_round = True
						)

		self.train_iterations = 0
		self.target_model = Model(layers = self._createLayers(num_actions))
		# for l in self.target_model.layers.layers:
		# 	l.parallelism = 'Disabled'
		self.target_model.initialize(self.input_shape[:-1])

		self.callback = None



	def _createLayers(self, num_actions):
		init_xavier_conv = Xavier(local=True)
		init_xavier_affine = Xavier(local=False)
		layers = []
		layers.append(Conv((8,8,32), strides=4, init=init_xavier_conv, activation=Rectlin(), batch_norm = self.batch_norm))
		layers.append(Conv((4,4,64), strides=2, init=init_xavier_conv, activation=Rectlin(), batch_norm = self.batch_norm))
		layers.append(Conv((2,2,128), strides=1, init=init_xavier_conv, activation=Rectlin(), batch_norm = self.batch_norm))
		layers.append(Affine(nout=256, init=init_xavier_affine, activation=Rectlin(), batch_norm = self.batch_norm))
		layers.append(Affine(nout=num_actions, init=init_xavier_affine))
		return layers

	def _setInput(self, states):
		states = np.transpose(states, axes = (1, 2, 3, 0))
		self.input.set(states.copy())
		self.be.add(self.input, 1, self.input)
		self.be.divide(self.input, 2, self.input)

	def update_target_network(self):
		pdict = self.model.get_description(get_weights=True, keep_states=True)
		self.target_model.deserialize(pdict, load_states=True)

	def train(self, minibatch, epoch):
		prestates, actions, rewards, poststates, terminals = minibatch

		self._setInput(poststates)
		postq = self.target_model.fprop(self.input, inference = True)
		assert postq.shape == (self.num_actions, self.batch_size)

		maxpostq = self.be.max(postq, axis=0).asnumpyarray()
		assert maxpostq.shape == (1, self.batch_size)

		self._setInput(prestates)
		preq = self.model.fprop(self.input, inference = False)
		assert preq.shape == (self.num_actions, self.batch_size)

		targets = preq.asnumpyarray().copy()
		rewards = np.clip(rewards, -1, 1)

		for i,action in enumerate(actions):
			if terminals[i]:
				targets[action, i] = float(rewards[i])
			else:
				targets[action, i] = float(rewards[i]) + self.discount_rate * maxpostq[0,i]

		self.targets.set(targets)

		deltas = self.cost.get_errors(preq, self.targets)
		assert deltas.shape == (self.num_actions, self.batch_size)

		cost = self.cost.get_cost(preq, self.targets)
		assert cost.shape == (1,1)

		if self.clip_error:
			self.be.clip(deltas, -self.clip_error, self.clip_error, out = deltas)

		self.model.bprop(deltas)
		self.optimizer.optimize(self.model.layers_to_optimize, epoch)

		self.train_iterations += 1
		self.callback.on_train(cost[0,0])


	def predict(self, states):
		assert states.shape == ((self.batch_size, self.history_length,) + self.board_dim)

		self._setInput(states)
		qvalues = self.model.fprop(self.input, inference = True)
		assert qvalues.shape == (self.num_actions, self.batch_size)

		return qvalues.T.asnumpyarray()

	def load_weights(self, load_path):
		self.model.load_params(load_path)

	def save_weights(self, save_path):
		self.model.save_params(save_path)


