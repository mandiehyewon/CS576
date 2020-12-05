
# import packages here
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

from scipy.misc import imresize  # resize images
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.svm import LinearSVC

category = ['aeroplane', 'car', 'horse', 'motorbike', 'person'] # DON'T MODIFY THIS.
data_dir = '/w11/hyewon/data/practical-category-recognition-2013a/data'

class_names = dict(zip(range(0,len(class_names)), class_names))
print (class_names)

def load_train_data(data_dir):
    dataset_setup(data_dir)
    num_proc = 12 # num_process

    txt_path = os.path.join(data_dir, 'train.txt')
    file_list = read_txt(txt_path)
    image_paths = [os.path.join(data_dir+'/images', file_name+'.jpg') for file_name in file_list]
    with multiprocessing.Pool(num_proc) as pool:
      imgs = pool.map(read_img, image_paths)
      imgs = np.array(imgs)
      idxs = np.array(file_list)

    return imgs, idxs

def load_val_data(data_dir):
    dataset_setup(data_dir)
    num_proc = 12 # num_process

    txt_path = os.path.join(data_dir, 'val.txt')
    file_list = read_txt(txt_path)
    image_paths = [os.path.join(data_dir+'/images', file_name+'.jpg') for file_name in file_list]
    with multiprocessing.Pool(num_proc) as pool:
      imgs = pool.map(read_img, image_paths)
      imgs = np.array(imgs)
      idxs = np.array(file_list)
    
    return imgs, idxs


# load training dataset
train_imgs, train_idxs = load_train_data(data_dir)
train_num = len(train_label)

# load testing dataset
test_data, test_label = load_val_data(data_dir)
test_num = len(test_label)

# feature extraction
def extract_feat(raw_data):
    feat_dim = 1000
    feat = np.zeros((len(raw_data), feat_dim), dtype=np.float32)
    for i in range(0,feat.shape[0]):
        feat[i] = np.reshape(raw_data[i], (raw_data[i].size))[:feat_dim] # dummy implemtation
        
    return feat

train_feat = extract_feat(train_data)
test_feat = extract_feat(test_data)

# model training: take feature and label, return model
def train(X, Y):
    return 0 # dummy implementation

# prediction: take feature and model, return label
def predict(model, x):
    return np.random.randint(15) # dummy implementation

# evaluation
predictions = [-1]*len(test_feat)
for i in range(0, test_num):
    predictions[i] = predict(None, test_feat[i])
    
accuracy = sum(np.array(predictions) == test_label) / float(test_num)

print ("The accuracy of my dummy model is {:.2f}%".format(accuracy*100))

def computeSIFT(data):
    x = []
    for i in range(0, len(data)):
        sift = cv2.xfeatures2d.SIFT_create()
        img = data[i]
        step_size = 15
        kp = [cv2.KeyPoint(x, y, step_size) for x in range(0, img.shape[0], step_size) for y in range(0, img.shape[1], step_size)]
        dense_feat = sift.compute(img, kp)
        x.append(dense_feat[1])
        
    return x

def extract_denseSIFT(img):
    DSIFT_STEP_SIZE = 2
    sift = cv2.xfeatures2d.SIFT_create()
    disft_step_size = DSIFT_STEP_SIZE
    keypoints = [cv2.KeyPoint(x, y, disft_step_size)
            for y in range(0, img.shape[0], disft_step_size)
                for x in range(0, img.shape[1], disft_step_size)]

    descriptors = sift.compute(img, keypoints)[1]
    
    #keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# build BoW presentation from SIFT of training images 
def clusterFeatures(all_train_desc, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(all_train_desc)
    return kmeans

# form training set histograms for each training image using BoW representation
def formTrainingSetHistogram(x_train, kmeans, k):
    train_hist = []
    for i in range(len(x_train)):
        data = copy.deepcopy(x_train[i])
        predict = kmeans.predict(data)
        train_hist.append(np.bincount(predict, minlength=k).reshape(1,-1).ravel())
        
    return np.array(train_hist)


# build histograms for test set and predict
def predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k):
    # form histograms for test set as test data
    test_hist = formTrainingSetHistogram(x_test, kmeans, k)
    
    # make testing histograms zero mean and unit variance
    test_hist = scaler.transform(test_hist)
    
    # Train model using KNN
    knn = trainKNN(train_hist, train_label, k)
    predict = knn.predict(test_hist)
    return np.array([predict], dtype=np.array([test_label]).dtype)
    

def accuracy(predict_label, test_label):
    return np.mean(np.array(predict_label.tolist()[0]) == np.array(test_label))

# extract dense sift features from training images
x_train = computeSIFT(train_data)
x_test = computeSIFT(test_data)

all_train_desc = []
for i in range(len(x_train)):
    for j in range(x_train[i].shape[0]):
        all_train_desc.append(x_train[i][j,:])

all_train_desc = np.array(all_train_desc)

k = [10, 15, 20, 25, 30, 35, 40]
for i in range(len(k)):
    kmeans = clusterFeatures(all_train_desc, k[i])
    train_hist = formTrainingSetHistogram(x_train, kmeans, k[i])
    
    # preprocess training histograms
    scaler = preprocessing.StandardScaler().fit(train_hist)
    train_hist = scaler.transform(train_hist)
    
    predict = predictKMeans(kmeans, scaler, x_test, train_hist, train_label, k[i])
    res = accuracy(predict, test_label)
    print("k =", k[i], ", Accuracy:", res*100, "%")

k = 60
kmeans = clusterFeatures(all_train_desc, k)

# form training and testing histograms
train_hist = formTrainingSetHistogram(x_train, kmeans, k)
test_hist = formTrainingSetHistogram(x_test, kmeans, k)

# normalize histograms
scaler = preprocessing.StandardScaler().fit(train_hist)
train_hist = scaler.transform(train_hist)
test_hist = scaler.transform(test_hist)

for c in np.arange(0.0001, 0.1, 0.00198):
    clf = LinearSVC(random_state=0, C=c)
    clf.fit(train_hist, train_label)
    predict = clf.predict(test_hist)
    print ("C =", c, ",\t Accuracy:", np.mean(predict == test_label)*100, "%")
