import numpy as np

train_ddes = np.load('train_ddes_1500.npy')
train_ddes1 = np.load('train_ddes_2000.npy')

dsift_d = np.vstack((train_ddes, train_ddes1))

print (dsift_d.shape)

del train_ddes, train_ddes1

# train_ddes = np.load('train_ddes_1500.npy')
# dsift_d = np.vstack((dsift_d, train_ddes))
# print (dsift_d.shape)
# del train_ddes

# train_ddes = np.load('train_ddes_2000.npy')
# dsift_d = np.vstack((dsift_d, train_ddes))
# print (dsift_d.shape)
# del train_ddes

train_ddes = np.load('train_ddes_final.npy')
dsift_d = np.vstack((dsift_d, train_ddes))
print (dsift_d.shape)
del train_ddes

try:
    np.save('./dsift_d.npy', dsift_d)
except:
    import pdb; pdb.set_trace()
import pdb; pdb.set_trace()