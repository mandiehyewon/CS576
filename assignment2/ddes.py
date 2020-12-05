import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cyvlfeat
import time
import scipy
import multiprocessing
from sklearn.svm import SVC
import pdb

from utils import euclidean_dist, read_img, read_txt, load_train_data, load_val_data, get_labels
from sift import SIFT_extraction, DenseSIFT_extraction, get_codebook, extract_features

category = ['aeroplane', 'car', 'horse', 'motorbike', 'person'] # DON'T MODIFY THIS.
data_dir = '/w11/hyewon/data/practical-category-recognition-2013a/data'

# feat_params = {'extractor': SIFT_extraction, 'num_codewords':1024, 'result_dir':os.path.join(data_dir,'sift_1024')}
# svm_params = {'C': 1, 'kernel': 'linear'}

train_imgs, train_idxs = load_train_data(data_dir)
print ('loaded dataset....')

train_ddes = DenseSIFT_extraction(train_imgs)
print (train_ddes.shape)

pdb.set_trace()

np.save('./train_ddes.npy', train_ddes)