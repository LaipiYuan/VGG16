import sys, os
sys.path.append(os.pardir)
import numpy as np 
from common.optimizer import *


class Trainer:
	def __init__(self, network, x_train, t_train, x_valid, t_valid, 
				 epochs=20, mini_batch_size=100,
				 optimizer='SGD', optimizer_param={'lr': 0.001},
				 evaluate_sample_num_per_epoch=None, verbose=True):
		self.x_train = x_train
		self.t_train = t_train
		self.x_valid = x_valid
		self.t_valid = t_valid
		self.network = network

		self.batch_size = mini_batch_size
		self.epochs = epochs
		self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch	# x_train[: evaluate_sample_num_per_epoch]
		self.verbose = verbose	# print information

		# optimizer
		optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
								'adagrad': AdaGrad, 'rmsprop': RMSprop, 'adam': Adam}
			# dictionaries can deliver keyword arguments with the **-operator.
			# optimizer = SGD(lr=0.01)
		self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)	# dictionaries can deliver keyword arguments with the **-operator

		self.train_size = x_train.shape[0]
		self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)	# 60000/100 = 600
		self.max_iter = int(epochs * self.iter_per_epoch)
		self.current_iter = 0
		self.current_epoch = 0

		self.train_loss_list = []
		self.train_acc_list = []
		self.valid_acc_list = []


	def train_step(self):
		# Data: 
			# Random batch
		batch_mask = np.random.choice(self.train_size, self.batch_size)
		x_batch = self.x_train[batch_mask]
		t_batch = self.t_train[batch_mask]

		# calculate Gradient: multi-Layers Network
			# ( Affine -> Batch Norm -> Activation -> Dropout -> ... -> Affine -> Softmax with Loss )
			# calculate gradient, loss function + weight decay: dW, db, dgamma, dbeta
		grads = self.network.gradient(x_batch, t_batch)
		# Update weight and bias: 
			# Optimizer
		self.optimizer.update(self.network.params, grads)

		loss = self.network.loss(x_batch, t_batch)
		self.train_loss_list.append(loss)
		if self.verbose:
			print("train loss: " + str(loss))

		if self.current_iter % self.iter_per_epoch == 0:
			self.current_epoch += 1

			x_train_sample, t_train_sample = self.x_train, self.t_train
			x_valid_sample, t_valid_sample = self.x_valid, self.t_valid
			if not self.evaluate_sample_num_per_epoch is None:
				t = self.evaluate_sample_num_per_epoch
				x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
				x_valid_sample, t_valid_sample = self.x_valid[:t], self.t_valid[:t]

			train_acc = self.network.accuracy(x_train_sample, t_train_sample)
			valid_acc = self.network.accuracy(x_valid_sample, t_valid_sample)
			self.train_acc_list.append(train_acc)
			self.valid_acc_list.append(valid_acc)

			if self.verbose:
				print("===== epoch:" + str(self.current_epoch) + ",\ttrain acc: " + str(train_acc) + ",\tvalid acc: " + str(valid_acc) + " =====")

		self.current_iter += 1


	def train(self):
		for i in range(self.max_iter):
			self.train_step()

			if self.current_iter % self.iter_per_epoch == 0:
				if not os.path.exists("Records"):
					os.makedirs("Records")
				filename = os.path.join("Records", "params_" + str(i) + ".pkl")
				self.network.save_params(file_name=filename)


		valid_acc = self.network.accuracy(self.x_valid, self.t_valid)

		if self.verbose:
			print("=============== Final Validation Accuracy ===============")
			print("validation accuracy: " + str(valid_acc) )


