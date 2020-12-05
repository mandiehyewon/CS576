import os
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import glob
#import cyvlfeat
import time
import scipy
import multiprocessing
from sklearn.svm import SVC
import pdb

from utils import euclidean_dist, read_img, read_txt, load_train_data, load_val_data, get_labels
from sift import SIFT_extraction, DkenseSIFT_extraction, get_codebook, extract_features
# from spatialpyramid import SpatialPyramid, SP_Trainer, SP_Test

# def Trainer(feat_params, svm_params):
	
# 	"""
# 	Train the SVM classifier.

# 	:param feat_params(dict): parameters for feature extraction.
# 		['extractor'](function pointer): function for extrat local descriptoers. (e.g. SIFT_extraction, DenseSIFT_extraction, etc)
# 		['num_codewords'](int):
# 		['result_dir'](str): Diretory to save codebooks & results.
		
# 	:param svm_params(dict): parameters for classifier training.
# 		['C'](float): Regularization parameter.
# 		['kernel'](str): Specifies the kernel type to be used in the algorithm.
	 
# 	:return(sklearn.svm.SVC): trained classifier
# 	"""
	
# 	extractor = feat_params['extractor']
# 	k = feat_params['num_codewords']
# 	result_dir = feat_params['result_dir']
	
# 	if not os.path.isdir(result_dir):
# 		os.mkdir(result_dir)
	
# 	print("Load the training data...")
# 	start_time = time.time()
# 	train_imgs, train_idxs = load_train_data(data_dir)
# 	print("{:.4f} seconds".format(time.time()-start_time))
	
# 	print("Extract the local descriptors...")
# 	start_time = time.time()
# 	train_des = extractor(train_imgs)
# 	np.save(os.path.join(result_dir, 'train_des.npy'), train_des)
# 	print("{:.4f} seconds".format(time.time()-start_time))
	
# 	del train_imgs
	
# 	print("Construct the bag of visual words...")
# 	start_time = time.time()
# 	codebook = get_codebook(train_des, k)
# 	np.save(os.path.join(result_dir, 'codebook.npy'), codebook)
# 	print("{:.4f} seconds".format(time.time()-start_time))

# 	print("Extract the image features...")
# 	start_time = time.time()	
# 	train_features = extract_features(train_des, codebook)
# 	np.save(os.path.join(result_dir, 'train_features.npy'), train_features)
# 	print("{:.4f} seconds".format(time.time()-start_time))

# 	del train_des, codebook
	
# 	print('Train the classifiers...')
# 	accuracy = 0
# 	models = {}
	
# 	for class_name in category:
# 		target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_train.txt'.format(class_name)))])
# 		target_labels = get_labels(train_idxs, target_idxs)
		
# 		models[class_name] = train_classifier(train_features, target_labels, svm_params)
# 		train_accuracy = models[class_name].score(train_features, target_labels) 
# 		print('{} zClassifier train accuracy:	{:.4f}'.format(class_name ,train_accuracy))
# 		accuracy += train_accuracy
	
# 	print('Average train accuracy: {:.4f}'.format(accuracy/len(category)))
# 	del train_features, target_labels, target_idxs

# 	return models

# def Test(feat_params, models):
# 	"""
# 	Test the SVM classifier.

# 	:param feat_params(dict): parameters for feature extraction.
# 		['extractor'](function pointer): function for extrat local descriptoers. (e.g. SIFT_extraction, DenseSIFT_extraction, etc)
# 		['num_codewords'](int):
# 		['result_dir'](str): Diretory to load codebooks & save results.
		
# 	:param models(dict): dict of classifiers(sklearn.svm.SVC)
# 	"""
	
# 	extractor = feat_params['extractor']
# 	k = feat_params['num_codewords']
# 	result_dir = feat_params['result_dir']
	
# 	print("Load the validation data...")
# 	start_time = time.time()
# 	val_imgs, val_idxs = load_val_data(data_dir)
# 	print("{:.4f} seconds".format(time.time()-start_time))
	
# 	print("Extract the local descriptors...")
# 	start_time = time.time()
# 	val_des = extractor(val_imgs)
# 	np.save(os.path.join(result_dir, 'val_des.npy'), val_des)
# 	print("{:.4f} seconds".format(time.time()-start_time))
	
	
# 	del val_imgs
# 	codebook = np.load(os.path.join(result_dir, 'codebook.npy'))
	
# 	print("Extract the image features...")
# 	start_time = time.time()	
# 	val_features = extract_features(val_des, codebook)
# 	np.save(os.path.join(result_dir, 'val_features.npy'), val_features)
# 	print("{:.4f} seconds".format(time.time()-start_time))

# 	del val_des, codebook

# 	print('Test the classifiers...')
# 	accuracy = 0
# 	for class_name in category:
# 		target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_val.txt'.format(class_name)))])
# 		target_labels = get_labels(val_idxs, target_idxs)
		
# 		val_accuracy = models[class_name].score(val_features, target_labels)
# 		print('{} Classifier validation accuracy:	{:.4f}'.format(class_name ,val_accuracy))
# 		accuracy += val_accuracy
	
# 	del val_features, target_idxs, target_labels

# 	print('Average validation accuracy: {:.4f}'.format(accuracy/len(category)))




# ''' 
# Set your data path for loading images & labels.
# Example) data_dir = '/gdrive/My Drive/data'
# '''
category = ['aeroplane', 'car', 'horse', 'motorbike', 'person'] # DON'T MODIFY THIS.
data_dir = '/w11/hyewon/data/practical-category-recognition-2013a/data'

# feat_params = {'extractor': SIFT_extraction, 'num_codewords':1024, 'result_dir':os.path.join(data_dir,'sift_1024')}
# svm_params = {'C': 1, 'kernel': 'linear'}

print("Load the training data...")
start_time = time.time()
train_imgs, train_idxs = load_train_data(data_dir)
print("{:.4f} seconds".format(time.time()-start_time))
"""
train_ddes = DenseSIFT_extraction(train_imgs)
print (train_ddes.shape)

pdb.set_trace()

np.save('./train_ddes.npy', train_ddes)

feat_params = {'extractor': DenseSIFT_extraction, 'num_codewords':1024, 'result_dir':os.path.join(data_dir,'dsift_1024')}
svm_params = {'C': 1, 'kernel': 'linear'}
# train_ddes = np.load('./train_ddes.npy')
"""
# train_des = np.load('./train_des.npy', allow_pickle=True)
# val_des = np.load('./val_des.npy', allow_pickle=True)
# codebook = np.load('./codebook.npy')

# def extract_features(des, codebook):
# 	"""
# 	Construct the Bag-of-visual-Words histogram features for images using the codebook.
# 	HINT: Refer to helper functions.

# 	:param des(numpy.array): Descriptors.	shape:[num_images, num_des_of_each_img, 128]
# 	:param codebook(numpy.array): Bag of visual words. shape:[k, 128]
# 	:return(numpy.array): Bag of visual words shape:[num_images, k]

# 	"""
# 	# YOUR CODE HERE
# 	bow_hist = []
# 	k = codebook.shape[0]
# 	for i in range(len(des)):
# 		dist = euclidean_dist(des[i], codebook) # [num_des_of_each_img, k]
# 		cluster_idx = np.argmin(dist, axis=1) 
# 		# print (cluster_idx.shape)
# 		cluter_idx = cluster_idx.reshape(-1) 
# 		# print (cluster_idx.shape)
# 		idx_onehot = np.eye(k)[cluster_idx]
# 		# print (idx_onehot.shape)
# 		bow_img = idx_onehot.sum(axis=0).reshape(-1, idx_onehot.sum(axis=0).shape[0])
# 		bow_hist.append(bow_img)
# 		# print (bow_img.shape)

# 	return np.array(bow_hist)


# train_bow = extract_features(train_des, codebook)
# np.save('./train_bow.npy', train_bow)
# print ('saved train_bow', train_bow.shape)

# valid_bow = extract_features(val_des, codebook)
# np.save('./valid_bow.npy', valid_bow)
# print ('saved valid_bow', valid_bow.shape)

# pdb.set_trace()
# codebook_d = get_codebook(train_ddes, k=1024)
# np.save('./codebook_d.npy', codebook_d)
# print (codebook.shape, codebook_d.shape)

# models = Trainer(feat_params, svm_params)
def train_classifier(features, labels, svm_params):
  """
  Train the SVM classifier using sklearn.svm.svc()
  Refer to https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

  :param features(numpy.array): Historgram representation. shape:[num_images, dim_feature]
  :param labels(numpy.array): Target label(binary). shape:[num_images,]
  :return(sklearn.svm.SVC): Trained classifier
  """
  # Your code here
  model = SVC(C=svm_params['C'], kernel = svm_params['kernel'])
  model.fit(features, labels)

  return model

# Test(feat_params ,models)
print ('load features, codebook')
train_features = np.load('./train_bow.npy')
val_features = np.load('./valid_bow.npy')

print('Train the classifiers...')
accuracy = 0
models = {}
svm_params = {'C': 1, 'kernel': 'linear'}

for class_name in category:
	target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_train.txt'.format(class_name)))])
	target_labels = get_labels(train_idxs, target_idxs)
	
	models[class_name] = train_classifier(train_features, target_labels, svm_params)
	train_accuracy = models[class_name].score(train_features, target_labels) 
	print('{} zClassifier train accuracy:	{:.4f}'.format(class_name ,train_accuracy))
	accuracy += train_accuracy

print('Average train accuracy: {:.4f}'.format(accuracy/len(category)))
del train_features, target_labels, target_idxs


print("Load the validation data...")
start_time = time.time()
val_imgs, val_idxs = load_val_data(data_dir)
print("{:.4f} seconds".format(time.time()-start_time))

print('Test the classifiers...')
accuracy = 0
for class_name in category:
	target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_val.txt'.format(class_name)))])
	target_labels = get_labels(val_idxs, target_idxs)
	
	val_accuracy = models[class_name].score(val_features, target_labels)
	print('{} Classifier validation accuracy:	{:.4f}'.format(class_name ,val_accuracy))
	accuracy += val_accuracy

del val_features, target_idxs, target_labels

print('Average validation accuracy: {:.4f}'.format(accuracy/len(category)))
