import numpy as np 


def smooth_curve(x):	
	# Used to smooth the graph of the loss function
	window_len = 11
	s = np.r_[ x[window_len-1: 0: -1], x, x[-1: -window_len: -1]]
	w = np.kaiser(window_len, 2)	# beta = 2
	y = np.convolve( w / w.sum(), s, mode='valid')
	return y[5: len(y)-5]

	

def shuffle_dataset(x, t):
	# Shuffle the data set
	permutation = np.random.permutation(x.shape[0])
	x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
	t = t[permutation]

	return x, t




def conv_output_size(input_size, filter_size, stride=1, pad=0):
	return (input_size + 2*pad - filter_size) / stride + 1



# image to column
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
	# input  shape: (batch_num, channel, height, width)
	# filter shape: (filter_num, channel, filter_height, filter_width)
	# output shape: (batch_num, filter_num, output_height, output_width)
	N, C, H, W = input_data.shape

	out_h = (H + 2*pad - filter_h) // stride + 1	# //: The division of operands where the result is the quotient in which the digits after the decimal point are removed.
	out_w = (W + 2*pad - filter_w) // stride + 1		# But if one of the operands is negative, the result is floored.

	# np.pad(array, pad_width, mode, **kwargs)
		# pad_width:
		# Number of values padded to the edges of each axis.
		# ((before_1, after_1), ... (before_N, after_N)) unique pad widths for each axis.
	img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')	# padding
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))	# 6-dim

	for y in range(filter_h):
		y_max = y + stride * out_h

		for x in range(filter_w):
			x_max = x + stride * out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

	col = col.transpose(0, 4, 5, 1, 2, 3).reshape( N*out_h*out_w, -1)

	'''  EASY version
	col = np.zeros((N, C, out_h, out_w, filter_h, filter_w))

	for y in range(out_h):
		y_max = y + stride * filter_h 	# 2, 3, 4

		for x in range(out_w):
			x_max = x + stride * filter_w 	# 2, 3, 4

			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	
	col = col.transpose(0, 2, 3, 1, 4, 5)		# -> (N, out_h, out_w, C, filter_h, filter_w)
	col = col.reshape(N*out_h*out_w, -1)
	'''
	return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_shape

	out_h = (H + 2*pad - filter_h) // stride + 1
	out_w = (W + 2*pad - filter_w) // stride + 1

	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	
	for y in range(filter_h):
		y_max = y + stride * out_h

		for x in range(filter_w):
			x_max = x + stride * out_w

			img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

	'''  EASY version
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 1, 2, 4, 5)	# -> (N, C, out_h, out_w, filter_h, filter_w)
	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

	for y in range(out_h):
		y_max = y + stride * filter_h

		for x in range(out_w):
			x_max = x + stride * filter_w

			img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]
	'''
	return img[:, :, pad:H + pad, pad:W + pad]


