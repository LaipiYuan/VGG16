import os
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import tables



def _convert_numpy(hdf5_path):
	hdf5_file = tables.open_file(hdf5_path, mode='r')

	dataset = {}
	dataset['train_image'] = np.array(hdf5_file.root.train_images[:])
	dataset['train_label'] = np.array(hdf5_file.root.train_labels[:])
	dataset['valid_image'] = np.array(hdf5_file.root.valid_images[:])
	dataset['valid_label'] = np.array(hdf5_file.root.valid_labels[:])

	hdf5_file.close()
	return dataset



def _change_one_hot_label(Y):
	labels_one_hot = np.zeros((Y.size, 12))
	for idx, row in enumerate(labels_one_hot):
		row[Y[idx]] = 1

	return labels_one_hot



def load_data(image_size=224, normalize=True, one_hot_label=False, hdf5_path='hdf5_file/dataset_s299.hdf5'):
	if not os.path.exists(hdf5_path):
		print("\nLoad Data Error!\n")
		return
	
	dataset = _convert_numpy(hdf5_path)

	if normalize == True :
		for key in ('train_image', 'valid_image'):
			dataset[key] = dataset[key].astype(np.float32)
			dataset[key] /= 255.0

	if one_hot_label:
		dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
		dataset['valid_label'] = _change_one_hot_label(dataset['valid_label'])		

	return (dataset['train_image'], dataset['train_label']), (dataset['valid_image'], dataset['valid_label'])
