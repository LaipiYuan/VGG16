import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np 
from collections import OrderedDict
from common.layers import *
'''
# weight_init_std  : He
# Layers: 16
# Optimizer        : Adam
'''
class VGG16:
	def __init__(self, input_dim=(3, 224, 224),
				 conv_param_1={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_2={'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_3={'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_4={'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_5={'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_6={'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_7={'filter_num':256, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_8={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_9={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_10={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_11={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_12={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 conv_param_13={'filter_num':512, 'filter_size':3, 'pad':1, 'stride':1},
				 hidden_size1=4096, hidden_size2=4096, output_size=1000):
		# weight initialization
			# Convolution
				# input shape : (N, C, H, W)
				# filter shape: (FN, C, FH, FW)		<- weight
				# output shape: (N, FN, C_OH, C_OW)
			# Pooling
				# input  shape: (N, FN, C_OH, C_OW)
				# pool   shape: (stride, stride)
				# output shape: (N, FN, P_OH, P_OW)
		pre_node_nums = np.array([3*3*3, 64*3*3, 64*3*3, 128*3*3, 128*3*3, 256*3*3, 256*3*3, 256*3*3,
								  512*3*3, 512*3*3, 512*3*3, 512*3*3, 512*3*3, 512*7*7,
								  hidden_size1, hidden_size2])	
			# n = (C * FH * FW)
		"""
				# Paper: y_l = W_l * x_l + b_l , l = convolution layer 
					# x is (k*k)*c-by-1 vecor, k×k pixels in c input channels. k is the spatial filter size of the layer.
						# x : (C * FH * FW, 1)
					# W is a d-by-n matrix, where d is the number of filters and each row of W represents the weights of a filter.
						# W : (FN, C * FH * FW)

				# Conv1, n=1*3*3;	Conv2, n=16*3*3; ...;	Conv6, n=64*3*3 (C=64, FS=3)
				# Affine7, n=64*4*4;	Affine8, n=50
		"""
		weight_init_scales = np.sqrt(2.0 / pre_node_nums)	# He

		self.params = {}
		pre_channel_num = input_dim[0]	# C
		for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5,
										  conv_param_6, conv_param_7, conv_param_8, conv_param_9, conv_param_10,
										  conv_param_11, conv_param_12, conv_param_13]):
			self.params['W' + str(idx+1)] = weight_init_scales[idx] *\
											np.random.randn( conv_param['filter_num'],
															 pre_channel_num,
															 conv_param['filter_size'],
															 conv_param['filter_size'] )
			self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])	# (FN, 1, 1)
			pre_channel_num = conv_param['filter_num']

		self.params['W14'] = weight_init_scales[13] * np.random.randn(512*7*7, hidden_size1)
		self.params['b14'] = np.zeros(hidden_size1)
		self.params['W15'] = weight_init_scales[14] * np.random.randn(hidden_size1, hidden_size2)
		self.params['b15'] = np.zeros(hidden_size2)
		self.params['W16'] = weight_init_scales[15] * np.random.randn(hidden_size2, output_size)
		self.params['b16'] = np.zeros(output_size)

		# Layers: 0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 34, 37
		self.layers = []
		# block 1
		self.layers.append( Convolution(self.params['W1'], self.params['b1'], 
										conv_param_1['stride'], conv_param_1['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W2'], self.params['b2'], 
										conv_param_2['stride'], conv_param_2['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Pooling(pool_h=2, pool_w=2, stride=2) )

		# block 2
		self.layers.append( Convolution(self.params['W3'], self.params['b3'], 
										conv_param_3['stride'], conv_param_3['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W4'], self.params['b4'], 
										conv_param_4['stride'], conv_param_4['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Pooling(pool_h=2, pool_w=2, stride=2) )

		# block 3
		self.layers.append( Convolution(self.params['W5'], self.params['b5'], 
										conv_param_5['stride'], conv_param_5['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W6'], self.params['b6'], 
										conv_param_6['stride'], conv_param_6['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W7'], self.params['b7'], 
										conv_param_7['stride'], conv_param_7['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Pooling(pool_h=2, pool_w=2, stride=2) )

		# block 4
		self.layers.append( Convolution(self.params['W8'], self.params['b8'], 
										conv_param_8['stride'], conv_param_8['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W9'], self.params['b9'], 
										conv_param_9['stride'], conv_param_9['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W10'], self.params['b10'], 
										conv_param_10['stride'], conv_param_10['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Pooling(pool_h=2, pool_w=2, stride=2) )

		# block 5
		self.layers.append( Convolution(self.params['W11'], self.params['b11'], 
										conv_param_11['stride'], conv_param_11['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W12'], self.params['b12'], 
										conv_param_12['stride'], conv_param_12['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Convolution(self.params['W13'], self.params['b13'], 
										conv_param_13['stride'], conv_param_13['pad']) )
		self.layers.append( Relu() )
		self.layers.append( Pooling(pool_h=2, pool_w=2, stride=2) )

		# hidden 1
		self.layers.append( Affine(self.params['W14'], self.params['b14']) )
		self.layers.append( Relu() )
		self.layers.append( Dropout(dropout_ratio=0.5) )

		# hidden 2
		self.layers.append( Affine(self.params['W15'], self.params['b15']) )
		self.layers.append( Relu() )
		self.layers.append( Dropout(dropout_ratio=0.5) )

		# output
		self.layers.append( Affine(self.params['W16'], self.params['b16']) )
		self.layers.append( Dropout(0.5) )

		self.last_layer = SoftmaxWithLoss()


	def predict(self, x, train_flg=False):
		for layer in self.layers:
			if isinstance(layer, Dropout):
				x = layer.forward(x, train_flg)
			else:
				x = layer.forward(x)

		return x


	def loss(self, x, t):
		y = self.predict(x, train_flg=True)
		return self.last_layer.forward(y, t)


	def accuracy(self, x, t, batch_size=10):
		if t.ndim != 1:		# one-hot vector
			t = np.argmax(t, axis=1)

		acc = 0.0
		for i in range( int(x.shape[0] / batch_size) ):
			tx = x[ i * batch_size : (i+1) * batch_size ]
			tt = t[ i * batch_size : (i+1) * batch_size ]
			y = self.predict(tx, train_flg=False)
			y = np.argmax(y, axis=1)
			
			acc += np.sum(y == tt)

		return acc / x.shape[0]


	def gradient(self, x, t):
		# forward
		self.loss(x, t)

		# backward
		dout = 1
		dout = self.last_layer.backward(dout)

		tmp_layers = self.layers.copy()
		tmp_layers.reverse()
		for layer in tmp_layers:
			dout = layer.backward(dout)

		grads = {}
		for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 34, 37)):
			grads['W' + str(i+1)] = self.layers[layer_idx].dW
			grads['b' + str(i+1)] = self.layers[layer_idx].db

		return grads


	def save_params(self, file_name="params.pkl"):
		params = {}
		for key, val in self.params.items():
			params[key] = val 

		with open(file_name, 'wb') as f:
			pickle.dump(params, f)


	def load_params(self, file_name="params.pkl"):
		with open(file_name, 'rb') as f:
			params = pickle.load(f)

		for key, val in params.items():
			self.params[key] = val

		for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28, 31, 34, 37)):
			self.layers[layer_idx].W = self.params['W' + str(i+1)]
			self.layers[layer_idx].b = self.params['b' + str(i+1)]



