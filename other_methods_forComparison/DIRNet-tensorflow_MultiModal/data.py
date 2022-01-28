import numpy as np
import os
import tensorflow as tf
from os import path, walk
from skimage import io


class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, is_train):
    self.is_train = is_train
    #self.path = path
    self.data, self.im_name = self._get_data()

  def _get_data(self):
    #from tensorflow.contrib.learn.python.learn.datasets.base \
    #  import maybe_download
    #from tensorflow.contrib.learn.python.learn.datasets.mnist \
    #  import extract_images, extract_labels

    #from tensorflow.examples.tutorials.mnist import input_data
    
    def get_file_names(dir, index_from=None, index_to=None):
	    for root, dirs, files in walk(dir):
		    return [path.join(dir, fn) for fn in files]


    dataset_name = '../Dl_Model_Perf_TestSet/Orig'  #'../Reg_Dataset_Breast_Original_all_patches_1024to256_set2'  #'Reg_Dataset_Breast_Original_all_patches_set2'   #'Dataset_256_10x_Grayscale'   #'sirius_data_patchwise_512'

    training_fixed_images_dir = path.join(dataset_name, 'Fixed')
    training_moving_images_dir = path.join(dataset_name, 'Moving')

    moving_images_fns = get_file_names(training_moving_images_dir)
    fixed_images_fns = get_file_names(training_fixed_images_dir)

    name_fixed1 = [i.replace('../Dl_Model_Perf_TestSet/Orig/Fixed/','') for i in fixed_images_fns]
    #name_moving1 = [i.replace('sirius_data_patchwise_512/Moving/','') for i in moving_images_fns]
    name_fixed = [i.replace('.png','') for i in name_fixed1]
    #name_moving = [i.replace('.png','') for i in name_moving1]

    moving_images = np.empty((len(moving_images_fns), 256, 256,3)) #1024, 1024, 3))  #256, 256, 3))
    for i in range(len(moving_images_fns)):
	    moving_images[i] = io.imread(moving_images_fns[i])
    
    fixed_images = np.empty((len(fixed_images_fns), 256, 256,3)) #1024, 1024, 3))  #256, 256, 3))
    for i in range(len(fixed_images_fns)):
	    fixed_images[i] = io.imread(fixed_images_fns[i])

    fixed_images = fixed_images.astype('float32')/255
    moving_images = moving_images.astype('float32')/255
   
    if self.is_train == True:
	    moving_images = moving_images[:]
	    fixed_images = fixed_images[:]
	    data = np.concatenate((moving_images, fixed_images), axis=0)
	    #data = np.expand_dims(data, axis=3)
	    im_name = name_fixed[:]
    else:
	    moving_images = moving_images[:]
	    fixed_images = fixed_images[:]
	    data = np.concatenate((moving_images, fixed_images), axis=0)
	    #data = np.expand_dims(data, axis=3)
	    im_name = name_fixed[:]

    #print(data)
    return data,im_name 

  def sample_pair(self, batch_size, label=None, start_idx = None, end_idx = None):

    num_images = int(self.data.shape[0] / 2)
    X1 = self.data[:num_images] 
    X2 = self.data[num_images:]

    if self.is_train == False:
	    x = X1[start_idx:end_idx+1]
	    y = X2[start_idx:end_idx+1]
	    name1 = self.im_name[start_idx:end_idx+1]
	    print(name1)
	    print(start_idx)
	    print(end_idx)

    else:
	    choice1 = np.random.choice(X1.shape[0], batch_size, replace=False)
	    x = X1[choice1]
	    y = X2[choice1]
	    choice2 = [int(i) for i in choice1]
	    print(choice2)
	    
	    name1 = [self.im_name[i] for i in choice2]
	    print(name1)

    #print(x)
    #print(y)

    #print(np.shape(x))
   

    return x, y, name1

