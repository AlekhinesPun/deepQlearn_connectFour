from itertools import groupby, chain
import random
import numpy as np


class Environment():
	def __init__(self,
					cols = 64,
					rows = 64,
					towin = 9,
					color = 1
				):
		self.cols = cols
		self.rows = rows
		self.towin = towin
		self.game = Game(self.cols,self.rows,self.towin)
		self.color = color

	def restart(self):
		self.game = Game(self.cols,self.rows,self.towin)

	def numActions(self):
		return self.cols

	def act(self, action):
		self.game.insert(action,self.color)
		if self.game.checkForWin() == self.color:
			return 1
		if self.game.checkFull():
			return 0
		self.game.randPlay(self.color*-1)
		if self.game.checkForWin() == self.color*-1:
			return -1*self.cols
		return 0

	def getBoard(self):
		return np.asarray(self.game.board)

	def isTerminal(self):
		win = self.game.checkForWin()
		ful = self.game.checkFull()
		if win or ful:
			return True





def diagonalsPos (matrix, cols, rows):
	"""Get positive diagonals, going from bottom-left to top-right."""
	for di in ([(j, i - j) for j in range(cols)] for i in range(cols + rows -1)):
		yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

def diagonalsNeg (matrix, cols, rows):
	"""Get negative diagonals, going from top-left to bottom-right."""
	for di in ([(j, i - cols + j + 1) for j in range(cols)] for i in range(cols + rows - 1)):
		yield [matrix[i][j] for i, j in di if i >= 0 and j >= 0 and i < cols and j < rows]

class Game:
	def __init__(self, cols=7, rows=7, requiredToWin=4):
		self.cols = cols
		self.rows = rows
		self.win = requiredToWin
		self.board = [[0] * rows for _ in range(cols)]

	def insert(self, column, color):
		c = self.board[column]
		if c[0] != 0:
			raise Exception('Column is full')
		i = -1
		while c[i] != 0:
			i-=1
		c[i] = color

	def checkForWin(self):
		lines = (
			self.board,
			zip(*self.board),
			diagonalsPos(self.board,self.cols,self.rows),
			diagonalsNeg(self.board,self.cols,self.rows)
		)

		for line in chain(*lines):
			for color, group in groupby(line):
				if color != 0 and len(list(group)) >= self.win:
					return color

	def randPlay(self,color):
		played = False
		while not played:
			try:
				action = random.randint(0,self.cols-1)
				self.insert(action,color)
				played = True
			except:
				played = False

	def checkFull(self):
		full = True
		for c in self.board:
			if c[0] == 0:
				full = False
		return full



