import os
from PIL import Image
import numpy as np
import glob
import multiprocessing

def euclidean_dist(x, y):
    """
    :param x: [m, d]
    :param y: [n, d]
    :return:[m, n]
    """
    m, n = x.shape[0], y.shape[0]    
    eps = 1e-6 

    xx = np.tile(np.power(x, 2).sum(axis=1), (n,1)) #[n, m]
    xx = np.transpose(xx) # [m, n]
    yy = np.tile(np.power(y, 2).sum(axis=1), (m,1)) #[m, n]
    xy = np.matmul(x, np.transpose(y)) # [m, n]
    dist = np.sqrt(xx + yy - 2*xy + eps)

    return dist

def read_img(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((480, 480))
    return np.float32(np.array(img)/255.)

def read_txt(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    return data.split()
    
def dataset_setup(data_dir):
    train_file_list = []
    val_file_list = []

    for class_name in ['aeroplane','background','car','horse','motorbike','person']:
        train_txt_path = os.path.join(data_dir, class_name+'_train.txt')
        train_file_list.append(np.array(read_txt(train_txt_path)))
        val_txt_path = os.path.join(data_dir, class_name+'_val.txt')
        val_file_list.append(np.array(read_txt(val_txt_path)))

    train_file_list = np.unique(np.concatenate(train_file_list))
    val_file_list = np.unique(np.concatenate(val_file_list))

    f = open(os.path.join(data_dir, "train.txt"), 'w')
    for i in range(train_file_list.shape[0]):
        data = "%s\n" % train_file_list[i]
        f.write(data)
    f.close()

    f = open(os.path.join(data_dir, "val.txt"), 'w')
    for i in range(val_file_list.shape[0]):
        data = "%s\n" % val_file_list[i]
        f.write(data)
    f.close()

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

def get_labels(idxs, target_idxs):
    """
    Get the labels from file index(name).

    :param idxs(numpy.array): file index(name). shape:[num_images, ]
    :param target_idxs(numpy.array): target index(name). shape:[num_target,]
    :return(numpy.array): Target label(Binary label consisting of True and False). shape:[num_images,]
    """
    return np.isin(idxs, target_idxs)