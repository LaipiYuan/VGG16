import os
import glob
import cv2
import numpy as np 


ROOT_DIR = "datasets/"
DST_DIR = "datasets_224/"


def create_image_lists(category):
	DATA_PATH = os.path.join(ROOT_DIR, category)

	result = {}		# key: class name, value: {train images, validation images}
	num_image = 0
	extensions = ['jpg', 'JPG', 'jpeg', 'JPEG', 'tif', "png"]

	if category == "train":
		sub_dirs = [x[0] for x in os.walk(DATA_PATH)]
		is_root_dir = True

		for sub_dir in sub_dirs:
			if is_root_dir:
				is_root_dir = False
				continue

			file_list = []
			dir_name = os.path.basename(sub_dir)
			for extension in extensions:
				file_glob = os.path.join(DATA_PATH, dir_name, '*.' + extension)
				file_list.extend(glob.glob(file_glob))	# glob.glob(pathname), 返回所有匹配的文件路徑列表。
			if not file_list: continue

			label_name = dir_name

			images = []
			for file_name in file_list:
				num_image += 1
				base_name = os.path.basename(file_name)
				images.append(base_name)

			result[label_name] = {'dir': dir_name,
								  'image': images}

	elif category == "test":
		file_list = []
		for extension in extensions:
			file_glob = os.path.join(DATA_PATH, '*.' + extension)
			file_list.extend(glob.glob(file_glob))

		images = []
		for file_name in file_list:
			num_image += 1
			base_name = os.path.basename(file_name)
			images.append(base_name)

			result = {'image': images}

	return result, num_image



def get_image_path(image_lists, label_name, index, category):
	DATA_PATH = os.path.join(ROOT_DIR, category)

	if category == "train":
		label_lists = image_lists[label_name]
		category_list = label_lists['image']
		base_name = category_list[index]
		sub_dir = label_lists['dir']	
		path = os.path.join(DATA_PATH, sub_dir, base_name)
	
	elif category == "test":
		category_list = image_lists['image']
		base_name = category_list[index]
		path = os.path.join(DATA_PATH, base_name)

	return path



def save_img(image_lists, category):
	DATA_PATH = os.path.join(ROOT_DIR, category)
	OUTPUT_DATA = '../datasets'

	if category == "train":
		label_name_list = list(image_lists.keys())
		for label_index, label_name in enumerate(label_name_list):
			dir_name = label_name

			for i in range(len(image_lists[label_name]['image'])):
				image_path = get_image_path(image_lists, label_name, i, category)
				img = cv2.imread(image_path)

				if img is not None:
					img_resize = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
				
				base_name = os.path.basename(image_path)
				save_path = os.path.join(DST_DIR, category, dir_name)
				
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				cv2.imwrite(os.path.join(save_path, base_name), img_resize, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
			
			print("File \"" + label_name + "\" is finished!----------- ", len(image_lists[label_name]['image']))
		print("\n***** Finished ALL *****\n")

	elif category == "test":
		for i in range(len(image_lists['image'])):
			image_path = get_image_path(image_lists, None, i, category)
			img = cv2.imread(image_path)

			if img is not None:
				img_resize = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
			
			base_name = os.path.basename(image_path)
			save_path = os.path.join(DST_DIR, category)
			
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			cv2.imwrite(os.path.join(save_path, base_name), img_resize, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
		
		print("\n***** Finished ALL *****\n")





def main():
	category = "train"
	img_size = 224

	image_lists, image_num = create_image_lists(category)
	print("image_num = ", image_num)

	save_img(image_lists, category)



if __name__ == '__main__':
	main()

