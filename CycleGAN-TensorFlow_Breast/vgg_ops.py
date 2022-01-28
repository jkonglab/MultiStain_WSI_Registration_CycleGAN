from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19


# For G generator

def _transfer_vgg19_weight_G(weight_dict):
	from_model = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256, 256, 3))
	
	fetch_weight = []
	for layer in from_model.layers:
		if 'conv' in layer.name:
			W, b = layer.get_weights()

			fetch_weight.append(tf.assign(weight_dict['G/{}/kernel'.format(layer.name)], W))

			fetch_weight.append(tf.assign(weight_dict['G/{}/bias'.format(layer.name)], b))
	
	return fetch_weight


def load_vgg19_weight_G():
	vgg_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
	assert len(vgg_weight) > 0, 'No VGG19 weight was collected. The target scope might be wrong.'
	weight_dict = {}
	for weight in vgg_weight:
		weight_dict[weight.name.rsplit(':', 1)[0]] = weight

	return _transfer_vgg19_weight_G(weight_dict)


### For F generator

def _transfer_vgg19_weight_F(weight_dict):
	from_model = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(256, 256, 3))

	fetch_weight = []
	for layer in from_model.layers:
		if 'conv' in layer.name:
			W, b = layer.get_weights()
			fetch_weight.append(tf.assign(weight_dict['F/{}/kernel'.format(layer.name)], W))
			fetch_weight.append(tf.assign(weight_dict['F/{}/bias'.format(layer.name)], b))

	return fetch_weight


def load_vgg19_weight_F():
	vgg_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='F')
	assert len(vgg_weight) > 0, 'No VGG19 weight was collected. The target scope might be wrong.'
	weight_dict = {}
	for weight in vgg_weight:
		weight_dict[weight.name.rsplit(':', 1)[0]] = weight

	return _transfer_vgg19_weight_F(weight_dict)


