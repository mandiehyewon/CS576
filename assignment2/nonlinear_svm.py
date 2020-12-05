import os
import numpy as np
import time
import scipy
from sklearn import svm

from utils import euclidean_dist, read_img, read_txt, load_train_data, load_val_data, get_labels

category = ['aeroplane', 'car', 'horse', 'motorbike', 'person'] # DON'T MODIFY THIS.
data_dir = '/w11/hyewon/data/practical-category-recognition-2013a/data'

train_imgs, train_idxs = load_train_data(data_dir)


def nonlinear_classifier(features, labels):
  model = svm.NuSVC(nu=1, gamma='auto')
  model.fit(features, labels)
  
  return model

def Nonlinear_Trainer():
    print("Load the training data...")
    start_time = time.time()
    train_imgs, train_idxs = load_train_data(data_dir)
    del train_imgs
    print("{:.4f} seconds".format(time.time()-start_time))

    print("Extract the image features...")
    train_features = np.load('./train_bow.npy')

    print('Train the classifiers...')
    accuracy = 0
    models = {}
    
    for class_name in category:
        target_idxs = np.array([read_txt(os.path.join(data_dir, '{}_train.txt'.format(class_name)))])
        target_labels = get_labels(train_idxs, target_idxs)
        
        models[class_name] = nonlinear_classifier(train_features, target_labels)
        train_accuracy = models[class_name].score(train_features, target_labels) 
        print('{} zClassifier train accuracy:  {:.4f}'.format(class_name ,train_accuracy))
        accuracy += train_accuracy
    
    print('Average train accuracy: {:.4f}'.format(accuracy/len(category)))
    del train_features, target_labels, target_idxs

    return models

def Nonlinear_Test(models):
    print("Load the validation data...")
    start_time = time.time()
    val_imgs, val_idxs = load_val_data(data_dir)
    print("{:.4f} seconds".format(time.time()-start_time))
    
    del val_imgs
    
    print("Extract the image features...")
    val_features = np.load('./val_bow.npy')

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



models = Nonlinear_Trainer()
Nonlinear_Test(models)