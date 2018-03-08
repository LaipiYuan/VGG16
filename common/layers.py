import numpy as np 
from functions import *
from util import im2col, col2im


class Relu:
	def __init__(self):
		self.mask = None

	def forward(self, x):
		self.mask = ( x <= 0 )
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0 
		dx = 1 * dout

		return dx



class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = sigmoid(x)
		self.out = out

		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out

		return dx 



class Affine:
	def __init__(self, W, b):
		self.W = W
		self.b = b

		self.x = None
		self.original_x_shape = None
		self.dW = None
		self.db = None

	def forward(self, x):
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x

		out = np.dot(self.x, self.W) + self.b
		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		dx = dx.reshape(*self.original_x_shape)	#  *-operator to unpack the arguments out of a list or tuple
		return dx



class SoftmaxWithLoss:
	def __init__(self):
		self.loss = None
		self.y = None
		self.t = None

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size:	# one-hot vector
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size

		return dx 



class BatchNormalization:
	# yi = gamma * normalize_xi + beta
	# normalize_xi = (xi - running_mean) / sqrt(running_variance + epsilon)
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None		# Convolution : 4 dimentions
									# other       : 2 dimentions

		self.running_mean = running_mean
		self.running_var = running_var

		# backward
		self.batch_size = None
		self.xc = None
		self.std = None

		self.dgamma = None
		self.dbeta = None

	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		if x.ndim != 2:				# tensor
			N, C, H, W = x.shape 	# N: batch, C: channels, H: height, W:, width
			x = x.reshape(N, -1)	# (batch, remaining dimentions)

		out = self.__forward(x, train_flg)

		return out.reshape(*self.input_shape)

	def __forward(self, x, train_flg):
		if self.running_mean is None:
			N, D = x.shape 					# x: (batch, Data)
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)

		if train_flg:
			mu = x.mean(axis=0)				# mean: (Data, )
			xc = x - mu
			var = np.mean(xc**2, axis=0)	# variance: (Data, )
			std = np.sqrt(var + 10e-7)		# standard deviation
			xn = xc / std					# normalize x

			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
		else:
			xc = x - self.running_mean
			xn = xc / ( (np.sqrt(self.running_var + 10e-7)) )

		out = self.gamma * xn + self.beta

		return out

	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
			dout = dout.reshape(N, -1)

		dx = self.__backward(dout)

		dx = dx.reshape(*self.input_shape)
		return dx 

	def __backward(self, dout):
		dbeta = dout.sum(axis=0)				# (D, )
		dgamma = np.sum(self.xn * dout, axis=0)	# (D, )
		dxn = self.gamma * dout					# (N, D)
		dxc = dxn / self.std 					# (N, D), dxc1
		dstd = -np.sum( (dxn * self.xc) / (self.std * self.std) , axis=0) 	# output = 1/std, (D, )
		dvar = 0.5 * dstd / self.std 	# output = std = sqrt(var), (D, )

		# output: sum(xc2^2), (N, D) 
		# dxc2 : dsquare = dvar * 1 / N
		#		 dxc2 = 2 * x * dsquare = (2 * xc * dvar) / N
		# node: minus, -> dxc1 + dxc2
		dxc += (2.0 / self.batch_size) * self.xc * dvar

		dmu = np.sum(dxc, axis=0)	# (D, )

		# dx = dx1 + dx2
		# dx1 = dxc 			(N, D)
		# dx2 = -dmu * 1 / N 	(N, D)
		dx = dxc - dmu / self.batch_size

		self.dgamma = dgamma
		self.dbeta = dbeta

		return dx



class Dropout:
	def __init__(self, dropout_ratio=0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:	# train
			# self.mask: bool, flase -> dropout
			# x.shape: tuple
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			return x * self.mask

		else:			# test: no dropout applied, but multiplying by ratio
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		return dout * self.mask




class Convolution:
	def __init__(self, W, b, stride=1, pad=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad

		# backward
		self.x = None
		self.col = None
		self.col_W = None

		self.dW = None
		self.db = None

	def forward(self, x):
		# input  shape: (batch_num, channel, height, width)
		# filter shape: (filter_num, channel, filter_height, filter_width)
		# output shape: (batch_num, filter_num, output_height, output_width)
		N, C, H, W = x.shape
		FN, C, FH, FW = self.W.shape

		out_h = 1 + int( (H + 2*self.pad - FH) / self.stride )	# OH
		out_w = 1 + int( (W + 2*self.pad - FW) / self.stride )	# OW

		col = im2col(x, FH, FW, self.stride, self.pad)	# (N * OH * OW,  C * FH * FW)
		col_W = self.W.reshape(FN, -1).T 				# (FN, C * FH * FW) -> (C * FH * FW, FN)

		out = np.dot(col, col_W) + self.b 				# (N * OH * OW, FN)
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)	# (N, FN, OH, OW)

		self.x = x
		self.col = col
		self.col_W = col_W

		return out

	def backward(self, dout):
		FN, C, FH, FW = self.W.shape
		dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)	# (N, OH, OW, FN) -> (N * OH * OW, FN)
		
		self.db = np.sum(dout, axis=0)		# (FN, 1, 1) ?
		self.dW = np.dot(self.col.T, dout)	# self.dW = np.dot(self.x.T, dout)
											# (C * FH * FW, N * OH * OW) * (N * OH * OW, FN) 
											# -> (C * FH * FW, FN)
		self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

		dcol = np.dot(dout, self.col_W.T)		# dx = np.dot(dout, self.W.T)
												# (N * OH * OW, FN) * (FN, C * FH * FW)
												# -> (N * OH * OW, C * FH * FW)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)	# (N, C, H, W)

		return dx

# Convolution
	# input shape : (N, C, H, W)
	# filter shape: (FN, C, FH, FW)
	# output shape: (N, FN, C_OH, C_OW)
# Pooling
	# input  shape: (N, FN, C_OH, C_OW)
	# pool   shape: (stride, stride)
	# output shape: (N, FN, P_OH, P_OW)
class Pooling:
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad

		self.x = None
		self.arg_max = None

	def forward(self, x):
		N, C, H, W = x.shape	## <- (N, FN, Conv_OH, Conv_OW)
		
		out_h = 1 + int( (H - self.pool_h) / self.stride )	## pooling OH -> OH
		out_w = 1 + int( (W - self.pool_w) / self.stride )	## pooling OW -> OW

		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)	# (N * OH * OW, C * PH * PW)
		col = col.reshape(-1, self.pool_h * self.pool_w)	# (N * OH * OW * C, PH * PW)

		arg_max = np.argmax(col, axis=1)	# Array of indices into the array, one-dimension
		out = np.max(col, axis=1)									# (N * OH * OW * C, 1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)	# (N, C, OH, OW)

		self.x = x
		self.arg_max = arg_max

		return out

	def backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1) 	# (N, OH, OW, C)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))	# (N * OH * OW * C, PH * PW)
		
		dmax[ np.arange(self.arg_max.size), self.arg_max.flatten() ] = dout.flatten()
			# np.arange(self.arg_max.size) : 0, 1, 2, ....., arg_max.size-1
			# self.arg_max.flatten()       : indices of the maximum values into arg_max, flattened to one dimension.
			# dout.flatten()               : (N * OH * OW * C, 1)
		dmax = dmax.reshape(dout.shape + (pool_size,))	# (N, OH, OW, C, PH * PW)
														# can only concatenate tuple

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)	# (N * OH * OW, C * PH * PW)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)	# (N, C, H, W)

		return dx





