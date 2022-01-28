import tensorflow as tf
import os
import skimage.io
import numpy as np
from scipy import ndimage
import keras.backend as K

def conv2d(x, name, dim, k, s, p, bn, af, is_train):
  with tf.variable_scope(name):
    w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
      initializer=tf.truncated_normal_initializer(stddev=0.01))
    x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

    if bn:
      x = batch_norm(x, "bn", is_train=is_train)
    else :
      b = tf.get_variable('biases', [dim],
        initializer=tf.constant_initializer(0.))
      x += b

    if af:
      x = af(x)

  return x


def reg(x, name, dim, k, s, p, is_train):
	with tf.variable_scope(name):
		w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
				                            initializer=tf.truncated_normal_initializer(stddev=0.01))

		x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
	return x


# conv2d_transpose filter: A 4D Tensor[height, width, output_channels, in_channels].
def conv2d_transpose(x, name, dim, output_shape, k, s, p, bn, af, is_train):
	with tf.variable_scope(name):
		w = tf.get_variable('weight', [k, k, dim, x.get_shape()[-1]],
				initializer=tf.truncated_normal_initializer(stddev=0.01))
		# output_shape = tf.constant([10, 8, 8, 64])
		x = tf.nn.conv2d_transpose(x, w, output_shape, [1, s, s, 1], p)
		if bn:
			x = batch_norm(x, "bn", is_train=is_train)
		else:
			b = tf.get_variable('biases', [dim], initializer=tf.constant_initializer(0.))
			x += b
		if af:
			x = af(x)
	return x

def reg(x, name, dim, k, s, p, is_train):
	with tf.variable_scope(name):
		w = tf.get_variable('weight', [k, k, x.get_shape()[-1], dim],
				initializer=tf.truncated_normal_initializer(stddev=0.01))
		x = tf.nn.conv2d(x, w, [1, s, s, 1], p)
	return x


def PReLU(_x, name=None):
	if name is None:
	    name = "alpha"
	_alpha = tf.get_variable(name, shape=_x.get_shape(), initializer=tf.constant_initializer(0.0), dtype=_x.dtype)
	return tf.maximum(_alpha*_x, _x)


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
  return tf.contrib.layers.batch_norm(x, 
    decay=momentum,
    updates_collections=None,
    epsilon=epsilon,
    scale=True,
    is_training=is_train, 
    scope=name)


def identity_matrix_init(shape, dtype=None, partition_info=None):
	            return np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()

def ncc(x, y):
  mean_x = tf.reduce_mean(x, [1,2,3], keep_dims=True)
  mean_y = tf.reduce_mean(y, [1,2,3], keep_dims=True)
  mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
  mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
  stddev_x = tf.reduce_sum(tf.sqrt(
    mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
  stddev_y = tf.reduce_sum(tf.sqrt(
    mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
  return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))



def gradientLoss(penalty='l1'):
	#scale = tf.constant([[par['res2']-1, 0, 0], [0, par['res1']-1, 0], [0,0,par['res3']-1]], dtype=tf.float32)
	def loss(y_pred):
		#y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)*0.5
		dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
		dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
		
		if (penalty == 'l2'):
			dy = dy * dy
			dx = dx * dx
		d = tf.reduce_mean(dx)+tf.reduce_mean(dy)   #+tf.reduce_mean(dz)
		return d/2.0
	
	return loss



def TvsLoss(penalty = 'l1', w1=1, w2=0):
#    scale = 0.5*tf.constant([[par['res2']-1.0, 0, 0], [0, par['res1']-1.0, 0], [0,0,par['res3']-1.0]], dtype=tf.float32)
        def loss(y_pred):
		#y_pred_O = tf.einsum('abcde,ef->abcdf',y_pred, scale)
                xx = y_pred[:,:,:,0] #Possible Problems here
                yy = y_pred[:,:,:,1]

                dy = tf.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
                dx = tf.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])

                if (penalty == 'l2'):
                        dy = dy * dy
                        dx = dx * dx

                        yy = yy * yy
                        xx = xx * xx

                d = tf.reduce_mean(dx)+tf.reduce_mean(dy)  #+tf.reduce_mean(dz)
                D = tf.reduce_mean(xx+yy)  #+zz)

                return w1*d/2.0+ w2*D
        return loss



def mse(x, y):
  #return tf.reduce_mean(tf.square(x - y))
  return tf.nn.l2_loss(x - y)


def mkdir(dir_path):
  try :
    os.makedirs(dir_path)
  except: pass 

def save_image_with_scale(path, arr):
  arr = np.clip(arr, 0., 1.)
  arr = arr * 255.
  arr = arr.astype(np.uint8)
  skimage.io.imsave(path, arr)


class DSSIMObjective:
	"""Computes DSSIM index between img1 and img2.
	This function is based on the standard SSIM implementation from:
	Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
	"""

	def __init__(self, k1=0.01, k2=0.03, max_value=1.0):
		self.__name__ = 'DSSIMObjective'
		self.k1 = k1
		self.k2 = k2
		self.max_value = max_value
		self.backend = K.backend()

	def __int_shape(self, x):
		return K.int_shape(x) if self.backend == 'tensorflow' else K.shape(x)

	def __call__(self, y_true, y_pred):
		ch = K.shape(y_pred)[-1]

		def _fspecial_gauss(size, sigma):
			#Function to mimic the 'fspecial' gaussian MATLAB function.
			coords = np.arange(0, size, dtype=K.floatx())
			coords -= (size - 1 ) / 2.0
			g = coords**2
			g *= ( -0.5 / (sigma**2) )
			g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
			g = K.constant ( np.reshape (g, (1,-1)) )
			g = K.softmax(g)
			g = K.reshape (g, (size, size, 1, 1))
			g = K.tile (g, (1,1,ch,1))
			return g
		kernel = _fspecial_gauss(11,1.5)
		def reducer(x):
			return K.depthwise_conv2d(x, kernel, strides=(1, 1), padding='valid')
		c1 = (self.k1 * self.max_value) ** 2
		c2 = (self.k2 * self.max_value) ** 2

		mean0 = reducer(y_true)
		mean1 = reducer(y_pred)
		num0 = mean0 * mean1 * 2.0
		den0 = K.square(mean0) + K.square(mean1)
		luminance = (num0 + c1) / (den0 + c1)

		num1 = reducer(y_true * y_pred) * 2.0
		den1 = reducer(K.square(y_true) + K.square(y_pred))
		c2 *= 1.0 #compensation factor
		cs = (num1 - num0 + c2) / (den1 - den0 + c2)

		ssim_val = K.mean(luminance * cs, axis=(-3, -2) )
		return K.mean( (1.0 - ssim_val ) / 2.0 )



EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
	"""
	Computes (normalized) mutual information between two 1D variate from a joint histogram.
	Parameters

	x : 1D array 
	    first variable
	y : 1D array 
	    second variable
	sigma: float
	       sigma for Gaussian smoothing of the joint histogram

	Returns
	nmi: float
	     the computed similariy measure
	"""

	bins = (20, 20)

	jh = np.histogram2d(x, y, bins=bins)[0]

	# smooth the jh with a gaussian filter of given sigma
	ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

	# compute marginal histograms
	jh = jh + EPS
	sh = np.sum(jh)
	jh = jh / sh
	s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
	s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

	# Normalised Mutual Information of:
	# Studholme,  jhill & jhawkes (1998).
	# "A normalized entropy measure of 3-D medical image alignment".
	# in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
	if normalized:
		mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
				/ np.sum(jh * np.log(jh))) - 1
	else:
		mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
				- np.sum(s2 * np.log(s2)))
	return mi


def convert2float(image):
	image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	return (image/127.5) - 1.0

def batch_convert2float(images):
	return tf.map_fn(convert2float, images, dtype=tf.float32)



