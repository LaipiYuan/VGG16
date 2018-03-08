import sys, os
sys.path.append(os.pardir)
import numpy as np 
import matplotlib.pyplot as plt 
#from dataset.data_reader import load_data
from data_reader import load_data
from vgg16 import VGG16
from trainer import Trainer


data_path = 'dataset/hdf5_file/dataset_s299_fold_4.hdf5'

(x_train, t_train), (x_valid, t_valid) = load_data(image_size=224, normalize=True, one_hot_label=False, hdf5_path=data_path)


network = VGG16()
trainer = Trainer(network, x_train, t_train, x_valid, t_valid,
				  epochs=20, mini_batch_size=5, 
				  optimizer='Adam', optimizer_param={'lr': 5e-5})

trainer.train()

network.save_params(file_name="Records/params.pkl")
print("\nSaved Network Parameters!\n")

