import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cyvlfeat
import time
import scipy
import multiprocessing

def SpatialPyramid(des, codebook):
	"""
	Extract image representation with Spatial Pyramid Matching using your DenseSIFT descripotrs & codebook.

	:param des(numpy.array): DenseSIFT Descriptors.	shape:[num_images, num_des_of_each_img, 128]
	:param codebook(numpy.array): Bag of visual words. shape:[k, 128]

	:return(numpy.array): Image feature using SpatialPyramid [num_images, features_dim]
	"""
	# YOUR CODE HERE

def SP_Trainer(des_path, codebook_path, result_dir, svm_params):
		
		"""
		Train the SVM classifier using SpatialPyramid representations.

		:param des_path(str): path for loading training dataset DenseSIFT descriptors.
		:param codebook(str): path for loading codebook for DenseSIFT descriptors.
		:param result_dir(str): diretory to save features.
				
		:param svm_params(dict): parameters for classifier training.
				['C'](float): Regularization parameter.
				['kernel'](str): Specifies the kernel type to be used in the algorithm.
	 
		:return(sklearn.svm.SVC): trained classifier
		"""
		train_des = np.load(des_path)
		codebook = np.load(codebook_path)
		train_features = SpatialPyramid(train_des, codebook)
		np.save(os.path.join(result_dir, 'train_sp_features.npy'), train_features)

		del train_des, codebook
		
		print('Train the classifiers...')
		accuracy = 0
		models = {}
		
		for class_name in category:
				target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_train.txt'.format(class_name)))])
				target_labels = get_labels(train_idxs, target_idxs)
				
				models[class_name] = train_classifier(train_features, target_labels, svm_params)
				train_accuracy = models[class_name].score(train_features, target_labels) 
				print('{} Classifier train accuracy:	{:.4f}'.format(class_name ,train_accuracy))
				accuracy += train_accuracy
		
		print('Average train accuracy: {:.4f}'.format(accuracy/len(category)))
		del train_features, target_labels, target_idxs

		return models

def SP_Test(des_path, codebook_path, result_dir, models):
	"""
	Test the SVM classifier.

	:param des_path(str): path for loading validation dataset DenseSIFT descriptors.
	:param codebook(str): path for loading codebook for DenseSIFT descriptors.
	:param result_dir(str): diretory to save features.	  
	:param models(dict): dict of classifiers(sklearn.svm.SVC)

	""" 
	val_des = np.load(des_path)
	codebook = np.load(codebook_path)
	val_features = SpatialPyramid(val_des, codebook)
	np.save(os.path.join(result_dir, 'val_sp_features.npy'), val_features)

	del val_des, codebook

	print('Test the classifiers...')
	accuracy = 0
	for class_name in category:
		target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_val.txt'.format(class_name)))])
		target_labels = get_labels(val_idxs, target_idxs)
		
		val_accuracy = models[class_name].score(val_features, target_labels)
		print('{} Classifier validation accuracy:  {:.4f}'.format(class_name ,val_accuracy))
		accuracy += val_accuracy

	del val_features, target_idxs, target_labels

	print('Average validation accuracy: {:.4f}'.format(accuracy/len(category)))
	