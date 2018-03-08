import os
import glob
import pickle
import numpy as np
from random import shuffle
import tables
import cv2
import csv
import pandas as pd
from tqdm import tqdm


data_order = 'th'
img_size = 224

if not os.path.exists("hdf5_file"):
	os.makedirs("hdf5_file")
hdf5_path = 'hdf5_file/dataset_s' + str(img_size) + '.hdf5'  # address to where you want to save the hdf5 file
csv_path = "hdf5_file/label_file_s" + str(img_size) + ".csv"

INPUT_DATA = "../datasets_224/train"

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved


label_dict = {'Motorola-Droid-Maxx'	: 0,
			  'Motorola-Nexus-6'	: 1,
			  'Samsung-Galaxy-Note3': 2,
			  'Motorola-X'			: 3,
			  'iPhone-4s'			: 4,
			  'iPhone-6'			: 5,
			  'Samsung-Galaxy-S4'	: 6,
			  'Sony-NEX-7'			: 7,
			  'LG-Nexus-5x'			: 8,
			  'HTC-1-M7'			: 9}


def write_data_hdf5(train_addrs, train_labels, valid_addrs, valid_labels, v):
	# check the order of data and chose proper data shape to save images
	if data_order == 'th':
		data_shape = (0, 3, img_size, img_size)
	elif data_order == 'tf':
		data_shape = (0, img_size, img_size, 3)

	# open a hdf5 file and create earrays
	hdf5_path = os.path.join('hdf5_file', 'dataset_s299_fold_' + str(v) + '.hdf5')
	hdf5_file = tables.open_file(hdf5_path, mode='w')
	# use "create_earray" which create an empty array
	train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_images', img_dtype, shape=data_shape)
	valid_storage = hdf5_file.create_earray(hdf5_file.root, 'valid_images', img_dtype, shape=data_shape)
	# create the label arrays and copy the labels data in them
	hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
	hdf5_file.create_array(hdf5_file.root, 'valid_labels', valid_labels)

	# loop over train addresses
	for i in range(len(train_addrs)):
		if i % 500 == 0 and i > 1:
			print('Train data: {}/{}'.format(i, len(train_addrs)))
		
		addr = train_addrs[i]
		img = cv2.imread(addr)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		if data_order == 'th':
			img = np.rollaxis(img, 2)	
		train_storage.append(img[None])

	# loop over validation addresses
	for i in range(len(valid_addrs)):
		if i % 50 == 0 and i > 1:
			print('Validation data: {}/{}'.format(i, len(valid_addrs)))

		addr = valid_addrs[i]
		img = cv2.imread(addr)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
		if data_order == 'th':
			img = np.rollaxis(img, 2)	
		valid_storage.append(img[None])

	hdf5_file.close()



def cross_validation(fold_num):  
	addrs = []
	labels = []
	datas = pd.read_csv(csv_path)
	for addr in tqdm(datas['addrs'].values):
		addrs.append(addr)
	for label in tqdm(datas['labels'].values):
		labels.append(label)

	data_len = len(addrs)	# 4750
	fold = int( data_len / fold_num )	# 4750/5 = 950
	addr_total = [addrs[i : i+fold] for i in range(0, data_len, fold)]
	label_total = [labels[i : i+fold] for i in range(0, data_len, fold)]

	for v in range(fold_num-1):
		train_addrs = []
		train_labels = []
		valid_addrs = []
		valid_labels = []

		addr_fold = addr_total.copy()
		label_fold = label_total.copy()

		valid_addrs = addr_fold.pop(v)
		valid_labels = label_fold.pop(v)

		for i in range(fold_num-1):
			train_addrs.extend(addr_fold[i]) 
			train_labels.extend(label_fold[i])

		write_data_hdf5(train_addrs, train_labels, valid_addrs, valid_labels, v)
		print(str(v) + "-fold is finished!")



def data_split(ratio_valid, shuffle_data, cross_valid, INPUT_DATA):
	sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
	addrs = []
	is_root_dir = True
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue

		extensions = ['jpg', 'JPG', 'tif', 'png']

		dir_name = os.path.basename(sub_dir)
		for extension in extensions:
			file_path = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
			addrs.extend(glob.glob(file_path))

	labels = []
	for addr in addrs:
		for key, value in label_dict.items():
			if key in addr:
				labels.append(value)

	# to shuffle data
	if shuffle_data:
		c = list(zip(addrs, labels))
		shuffle(c)
		addrs, labels = zip(*c)

		with open(csv_path, 'w', newline='') as csv_out_file:
			filewriter = csv.writer(csv_out_file, lineterminator='\n')
			filewriter.writerow(['addrs', 'labels'])
			filewriter.writerows(c)

	# Divide the hata into 90% train, 10% validation
	train_addrs = addrs[0 : int((1-ratio_valid) * len(addrs))]
	train_labels = labels[0 : int((1-ratio_valid) * len(labels))]
	valid_addrs = addrs[int((1-ratio_valid) * len(addrs)) :]
	valid_labels = labels[int((1-ratio_valid) * len(addrs)) :]

	fold_num = int(1. / ratio_valid)
	write_data_hdf5(train_addrs, train_labels, valid_addrs, valid_labels, fold_num-1)
	print("Write File is finished!")

	if cross_valid:
		cross_validation(fold_num)




def main():

	ratio_valid = 0.2
	cross_valid = False
	shuffle_data = True

	data_split(ratio_valid, shuffle_data, cross_valid, INPUT_DATA)



if __name__ == '__main__':
	main()

