import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cyvlfeat
import time
import scipy
import multiprocessing

from utils import euclidean_dist, read_img, read_txt, load_train_data, load_val_data, get_labels

def SIFT_extraction(imgs):
	"""
	Extract Local SIFT descriptors from images using cyvlfeat.sift.sift().
	Refer to https://github.com/menpo/cyvlfeat
	You should set the parameters of cyvlfeat.sift.sift() as below.
	1.compute_descriptor = True	2.float_descriptors = True

	:param train_imgs(numpy.array): Gray-scale images in Numpy array format. shape:[num_images, width_size, height_size]
	:return(numpy.array): SIFT descriptors. shape:[num_images, ], ndarray with object(descripotrs)
	"""
	sift_d = []
	num_images = imgs.shape[0]

	for i in range(num_images):
		image = np.squeeze(imgs[i])
		sift_d.append(cyvlfeat.sift.sift(image, float_descriptors=True,compute_descriptor=True)[:][1])

	return np.asarray(sift_d)

def DenseSIFT_extraction(imgs):
	"""
	Extract Dense SIFT descriptors from images using cyvlfeat.sift.dsift().
	Refer to https://github.com/menpo/cyvlfeat
	You should set the parameters of cyvlfeat.sift.dsift() as bellow.
		1.step = 12	2.float_descriptors = True

	:param train_imgs(numpy.array): Gray-scale images in Numpy array format. shape:[num_images, width_size, height_size]
	:return(numpy.array): Dense SIFT descriptors. shape:[num_images, num_des_of_each_img, 128]
	"""
	# YOUR CODE HERE
	
	dsift_d = cyvlfeat.sift.dsift(imgs[0], float_descriptors=True)[:][1]
	dsift_d = np.expand_dims(dsift_d, axis=0)
	print (dsift_d.shape)

	num_images = imgs.shape[0]

	for i in range(1001,num_images):
		#import pdb; pdb.set_trace()
		image = np.squeeze(imgs[i])
		new_dsift = cyvlfeat.sift.dsift(image, float_descriptors=True)[:][1]
		new_dsift = np.expand_dims(new_dsift, axis=0)
		# import pdb; pdb.set_trace()
		if i % 500 == 1:
			dsift_d = new_dsift
		else:
			dsift_d = np.vstack((dsift_d, new_dsift))
		print (new_dsift.shape, dsift_d.shape)

		if i % 500 == 0:
			np.save('./train_ddes_{}.npy'.format(i), dsift_d)
			del dsift_d

	return dsift_d

	# dsift_d = []
	# num_images = imgs.shape[0]

	# for i in range(num_images):
	# 	image = np.squeeze(imgs[i])
	# 	dsift_d.append(cyvlfeat.sift.dsift(image, float_descriptors=True)[:][1])

	# return np.array(dsift_d)
	
def get_codebook(des , k):
	"""
	Construct the codebook with visual codewords using k-means clustering.
	In this step, you should use cyvlfeat.kmeans.kmeans().
	Refer to https://github.com/menpo/cyvlfeat

	:param des(numpy.array): Descriptors.	shape:[num_images, num_des_of_each_img, 128]
	:param k(int): Number of visual words.
	:return(numpy.array): Bag of visual words shape:[k, 128]
	"""
	# YOUR CODE HERE
	all_desc = []
	for i in range(len(des)):
		for j in range(des[i].shape[0]):
			all_desc.append(des[i][j,:])
	all_desc = np.array(all_desc)

	codebook = cyvlfeat.kmeans.kmeans(all_desc, k)

	return codebook

def extract_features(des, codebook):
  """
  Construct the Bag-of-visual-Words histogram features for images using the codebook.
  HINT: Refer to helper functions.

  :param des(numpy.array): Descriptors.  shape:[num_images, num_des_of_each_img, 128]
  :param codebook(numpy.array): Bag of visual words. shape:[k, 128]
  :return(numpy.array): Bag of visual words shape:[num_images, k]

  """
  # YOUR CODE HERE
  bow_hist = []
  k = codebook.shape[0]
  for i in range(len(des)):
    # data = copy.deepcopy(des[i])
    dist = euclidean_dist(des[i], codebook)
    cluster_idx = np.argmin(dist, axis=1)

    targets = cluster_idx.reshape(-1)
    one_hot_targets = np.eye(k)[targets]
    bow_hist.append(onehot_targets)

  return np.array(bow_hist)

# features = np.zeros((2472, 1024))

# dist = cdist(obs, code_book)
# code = dist.argmin(axis=1)
# min_dist = dist[np.arange(len(code)), code]
# return code, min_dist