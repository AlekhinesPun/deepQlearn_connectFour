from env import Environment
from mem import ReplayMemory
from dqn import DeepQNetwork
from agt import Agent
from sta import Statistics
import random

env = Environment()
mem = ReplayMemory()
net = DeepQNetwork(env.numActions())
agt = Agent(env, mem, net)
stats = Statistics(agt, net, mem, env)
stats.reset()

for epoch in xrange(0,10):
	agt.train(2500, epoch)
	stats.out('train',epoch+1)
	agt.test(1250, epoch)
	stats.out('test',epoch+1)
