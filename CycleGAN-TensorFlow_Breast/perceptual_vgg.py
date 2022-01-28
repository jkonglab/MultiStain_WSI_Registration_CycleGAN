import tensorflow as tf
import ops

class Perceptual_VGG19:
	def __init__(self, name, is_training):
		self.name = name
		self.is_training = is_training
		self.reuse = tf.AUTO_REUSE

	def __call__(self, input):
	    with tf.variable_scope(self.name):
                
                x = ops.dk_s1(input, 64, reuse=self.reuse, norm=None, is_training=self.is_training, name='block1_conv1')    
		#x = tf.layers.conv2d(input, 64, (3, 3), activation='relu', padding='same', name='block1_conv1')
                x = ops.dk_s1(x, 64, reuse=self.reuse, norm=None, is_training=self.is_training, name='block1_conv2')
		#x = tf.layers.conv2d(x, 64, (3, 3), activation='relu', padding='same', name='block1_conv2')
                #x = ops.dk_pool(x, reuse=self.reuse, is_training=self.is_training, name='block1_pool')
                x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block1_pool')
		
		# Block 2
                x = ops.dk_s1(x, 128, reuse=self.reuse, norm=None, is_training=self.is_training, name='block2_conv1')
                #x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv1')
                
                x = ops.dk_s1(x, 128, reuse=self.reuse, norm=None, is_training=self.is_training, name='block2_conv2')
		#x = tf.layers.conv2d(x, 128, (3, 3), activation='relu', padding='same', name='block2_conv2')
                #x = ops.dk_pool(x, reuse=self.reuse,  is_training=self.is_training, name='block2_pool')
                x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block2_pool')

		# Block 3
                x = ops.dk_s1(x, 256, reuse=self.reuse, norm=None, is_training=self.is_training, name='block3_conv1')
		#x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv1')
                x = ops.dk_s1(x, 256, reuse=self.reuse, norm=None, is_training=self.is_training, name='block3_conv2')
		#x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv2')
                x = ops.dk_s1(x, 256, reuse=self.reuse, norm=None, is_training=self.is_training, name='block3_conv3')

		#x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv3')
                
                x = ops.dk_s1(x, 256, reuse=self.reuse, norm=None, is_training=self.is_training, name='block3_conv4')
		#x = tf.layers.conv2d(x, 256, (3, 3), activation='relu', padding='same', name='block3_conv4')
		
                #x = ops.dk_pool(x, reuse=self.reuse,  is_training=self.is_training, name='block3_pool')
                x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block3_pool')

		# Block 4

                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block4_conv1')
	
		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv1')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block4_conv2')
		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv2')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block4_conv3')
		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv3')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block4_conv4')
		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block4_conv4')
                #x = ops.dk_pool(x, reuse=self.reuse, is_training=self.is_training, name='block4_pool')
                x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2), name='block4_pool')

		# Block 5
                
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block5_conv1')

		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv1')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block5_conv2')
                #x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv2')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block5_conv3')
		#x = tf.layers.conv2d(x, 512, (3, 3), activation='relu', padding='same', name='block5_conv3')
                x = ops.dk_s1(x, 512, reuse=self.reuse, norm=None, is_training=self.is_training, name='block5_conv4')
		#x = tf.layers.conv2d(x, 512, (3, 3), activation=None, padding='same', name='block5_conv4')

                return x

