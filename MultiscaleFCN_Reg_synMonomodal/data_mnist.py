import numpy as np
import os
import tensorflow as tf

class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, path, is_train):
    self.is_train = is_train
    self.path = path
    self.data = self._get_data()

  def _get_data(self):
    #from tensorflow.contrib.learn.python.learn.datasets.base \
    #  import maybe_download
    #from tensorflow.contrib.learn.python.learn.datasets.mnist \
    #  import extract_images, extract_labels

    #from tensorflow.examples.tutorials.mnist import input_data
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    #print(x_test)
    #print(y_test)
    if self.is_train:
      
      IMAGES = x_train  #'train-images-idx3-ubyte.gz'
      LABELS = y_train  #'train-labels-idx1-ubyte.gz'
    else :
      IMAGES = x_test  #'t10k-images-idx3-ubyte.gz'
      LABELS = y_test  #'t10k-labels-idx1-ubyte.gz'
    
    #SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    #local_file = maybe_download(IMAGES, self.path, SOURCE_URL)
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


    #local_file = os.path.join(self.path, IMAGES)
    #with open(local_file, 'rb') as f:
    #  images = extract_images(f)
    #local_file = maybe_download(LABELS, self.path, SOURCE_URL)
    #with open(local_file, 'rb') as f:
    #  labels = extract_labels(f, one_hot=False)

    images = IMAGES
    labels = LABELS
    #print(np.shape(images))
    images = np.expand_dims(images, axis=3)
    
    #print(np.shape(images))

    #print(np.shape(labels))
    values, counts = np.unique(labels, return_counts=True)

    data = []
    for i in range(10):
      label = values[i]
      count = counts[i]
      arr = np.empty([count, 28, 28, 1], dtype=np.float32)
      print(np.shape(arr))
      data.append(arr)
      
    #print(data)

    l_iter = [0]*10
    for i in range(labels.shape[0]):
      label = labels[i]
      data[label][l_iter[label]] = images[i]  #/ 255.
      l_iter[label] += 1
    
    #print(np.shape(data))
    #print(data)
    return data

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(10) if label is None else label
    images = self.data[label]
    print(np.shape(images)) 

    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]


    #print(np.shape(x))
    #print(np.shape(y))
    return x, y

