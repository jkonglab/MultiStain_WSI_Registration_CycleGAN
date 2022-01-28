import tensorflow as tf
from WarpST import WarpST
from ops import *
from AffineST import transformer
from bicubic_interp import bicubic_interp_2d
from bicubic_downsample import build_filter, apply_bicubic_downsample


class CNN_scale1(object):
  def __init__(self, name, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
  
  def __call__(self, x):
    with tf.variable_scope(self.name, reuse=self.reuse):
      x_1 = conv2d(x, 'Conv1', 32, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_2 = tf.nn.avg_pool(x_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
      x_3 = conv2d(x_2, 'Conv2', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_4 = tf.nn.avg_pool(x_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
      x_5 = conv2d(x_4, 'Conv3', 128, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_6 = conv2d_transpose(x_5, 'deconv1', 128, [1, 16, 16, 128], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_7 = conv2d(x_6, 'Conv4', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_8 = conv2d_transpose(x_7, 'deconv2', 64, [1, 16, 16, 64], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
      x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self.is_train)
      x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self.is_train)
      x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self.is_train)

      """
      """

    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True

    return x_9   #, x_10, x_11  #z_256, z_128, z_64      #z_512, z_256, z_128  #x

  def save(self, sess, ckpt_path):
    self.saver.save(sess, ckpt_path)

  def restore(self, sess, ckpt_path):
    self.saver.restore(sess, ckpt_path)


class CNN_scale2(object):
	def __init__(self, name, is_train):
		self.name = name
		self.is_train = is_train
		self.reuse = None
	def __call__(self, x):
		with tf.variable_scope(self.name, reuse=self.reuse):
			x_1 = conv2d(x, 'Conv1', 32, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_2 = tf.nn.avg_pool(x_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
			x_3 = conv2d(x_2, 'Conv2', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_4 = tf.nn.avg_pool(x_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
			x_5 = conv2d(x_4, 'Conv3', 128, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_6 = conv2d_transpose(x_5, 'deconv1', 128, [1, 32, 32, 128], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_7 = conv2d(x_6, 'Conv4', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_8 = conv2d_transpose(x_7, 'deconv2', 64, [1, 32, 32, 64], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self.is_train)
			x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self.is_train)
			x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self.is_train)
		
		if self.reuse is None:
			self.var_list = tf.get_collection(
					        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
			self.saver = tf.train.Saver(self.var_list)
			self.reuse = True

		return x_9
	def save(self, sess, ckpt_path):
		self.saver.save(sess, ckpt_path)
	def restore(self, sess, ckpt_path):
		self.saver.restore(sess, ckpt_path)


class CNN_scale3(object):
	def __init__(self, name, is_train):
		self.name = name
		self.is_train = is_train
		self.reuse = None
	def __call__(self, x):
		with tf.variable_scope(self.name, reuse=self.reuse):
			x_1 = conv2d(x, 'Conv1', 32, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_2 = tf.nn.avg_pool(x_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling1')
			x_3 = conv2d(x_2, 'Conv2', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_4 = tf.nn.avg_pool(x_3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='pooling2')
			x_5 = conv2d(x_4, 'Conv3', 128, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_6 = conv2d_transpose(x_5, 'deconv1', 64, [1, 64, 64, 64], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_7 = conv2d(x_6, 'Conv4', 64, 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_8 = conv2d_transpose(x_7, 'deconv2', 32, [1, 64, 64, 32], 3, 1, 'SAME', True, tf.nn.relu, self.is_train)
			x_9 = reg(x_8, 'Reg1', 2, 3, 1, 'SAME', self.is_train)
			x_10 = reg(x_7, 'Reg2', 2, 3, 1, 'SAME', self.is_train)
			x_11 = reg(x_5, 'Reg3', 2, 3, 1, 'SAME', self.is_train)

		if self.reuse is None:
			self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
			self.saver = tf.train.Saver(self.var_list)
			self.reuse = True

		return x_9, x_10, x_11
	def save(self, sess, ckpt_path):
		self.saver.save(sess, ckpt_path)
	def restore(self, sess, ckpt_path):
		self.saver.restore(sess, ckpt_path)



class DIRNet(object):
  def __init__(self, sess, config, name, is_train):
    self.sess = sess
    self.name = name
    self.is_train = is_train

    # moving / fixed images
    im_shape = [config.batch_size] + config.im_size + [3]
    self.x = tf.placeholder(tf.float32, shape = im_shape)  # moving image
    self.y = tf.placeholder(tf.float32, shape = im_shape)  # ref image
    self.xy = tf.concat([self.x, self.y], axis = 3)

    #self.sess.run(print((self.xy).shape))
    ##  Scale 1 ####
    # downsampled to fcator 4
    #kk = build_filter(factor=4)
    #self.xy_down4 = apply_bicubic_downsample(self.xy, filter=kk, factor=4)
    self.xy_down4 = bicubic_interp_2d(self.xy,[64, 64])
    #self.xy_down4 = tf.image.resize_images(self.xy, size=(64, 64))
    #self.xy_down4 = batch_convert2float(self.xy_down4)

    self.vCNN1 = CNN_scale1("vector_CNN1", is_train=self.is_train)

    self.v_down4 = self.vCNN1(self.xy_down4)  # dvf at scale 1
    self.z_scale1  = WarpST(self.x, self.v_down4, config.im_size)
    ###########################
    ## Scale 2######
    self.z1y = tf.concat([self.z_scale1, self.y],axis= 3)
    
    #kkk = build_filter(factor=2)
    #self.z1y_down2 = apply_bicubic_downsample(self.z1y, filter=kkk, factor=2)
    #z1y_down2 = downsample(self.z1y)
    self.z1y_down2 = bicubic_interp_2d(self.z1y, [128, 128])
    
    #self.xy_down2 = tf.image.resize_images(self.xy, size=(128, 128))
    #self.xy_down2 = batch_convert2float(self.xy_down2)
    
    self.vCNN2 = CNN_scale2("vector_CNN2", is_train=self.is_train)

    self.v_down2 = self.vCNN2(self.z1y_down2)  # residual dvf at scale 2
    
    #self.V_down4 = bicubic_interp_2d(self.v_down4, config.im_size)
    #self.V_down2res = bicubic_interp_2d(self.v_down2, config.im_size)

    #self.V_down2 = self.V_down4 + self.V_down2res

    self.z_scale2  = WarpST(self.x, self.v_down2, config.im_size)
    ###############################
    ## Scale 3######

    self.z2y = tf.concat([self.z_scale2, self.y], axis=3)

    self.vCNN3 = CNN_scale3("vector_CNN3", is_train=self.is_train)

    self.vres = self.vCNN3(self.xy)   #self.z2y)

    self.vres1 = self.vres[0]
    self.vres2 = self.vres[1]
    self.vres3 = self.vres[2]
   
    #self.V_down2 = bicubic_interp_2d(self.v_down2, config.im_size)
    #self.Vres = bicubic_interp_2d(self.vres1, config.im_size)

    #self.V = self.Vres + self.V_down2
    self.z1  = WarpST(self.x, self.vres1, config.im_size)

    self.z2 = WarpST(self.x, self.vres2, config.im_size)
    self.z3 = WarpST(self.x, self.vres3, config.im_size)

    ################################

    # calculate loss
    #self.loss1 = -ncc(self.y, self.u) + (-ncc(self.y, self.z1)) + TvsLoss('l2')(self.v) + gradientLoss('l2')(self.v) 

    #self.loss1 = -ncc(self.y, self.z)
    self.loss_scale31 = -ncc(self.y, self.z1)
    self.loss_scale32 = -ncc(self.y, self.z2)
    self.loss_scale33 = -ncc(self.y, self.z3)

    self.loss_scale2 = -ncc(self.y, self.z_scale2)
    self.loss_scale1 = -ncc(self.y, self.z_scale1)
    
    self.loss4 = gradientLoss('l2')(self.vres1)/490
    self.loss5 = gradientLoss('l2')(self.vres2)/90
    self.loss6 = gradientLoss('l2')(self.vres3)/90

    #self.loss = #(0.05*self.loss_scale1 +
    
    self.loss = 0.05*self.loss_scale1 + 0.05*self.loss_scale2 + 0.9 * self.loss_scale31 + self.loss4  + 0.6 * (self.loss_scale32 +self.loss5) + 0.3 * (self.loss_scale33 + self.loss6)
    

    #Compute all different kind of losses (for quantitative metric)
    self.loss_ncc = -ncc(self.y, self.z1)
    #self.loss_NMI = mutual_information_2d(np.ravel(self.y), np.ravel(self.z1))
    loss_DSSIM = DSSIMObjective()
    self.loss_ssim = loss_DSSIM(self.y, self.z1)
    self.loss_mse = mse(self.y, self.z1)
    

    if self.is_train :
      global_step = tf.Variable(0, trainable=False)
      decayed_lr = tf.train.exponential_decay(config.lr, global_step, 5000, 0.96, staircase=True)
      self.optim = tf.train.AdamOptimizer(decayed_lr)  #config.lr)
      self.train = self.optim.minimize(self.loss, var_list=[self.vCNN1.var_list, self.vCNN2.var_list,self.vCNN3.var_list], global_step=global_step)

    #self.sess.run(
    #  tf.variables_initializer(self.vCNN.var_list))
    self.sess.run(
      tf.global_variables_initializer())


  def fit(self, batch_x, batch_y):
    _, loss = \
      self.sess.run([self.train, self.loss], 
      {self.x:batch_x, self.y:batch_y})
    return loss #, loss1, loss2, loss3, loss4, loss5, loss6

  def deploy(self, dir_path, x, y, name1):
    z1  = self.sess.run(self.z1, {self.x:x, self.y:y})

    
    #u  = self.sess.run(self.u, {self.x:x, self.y:y})
    loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y: y})
    
    loss_nmi = mutual_information_2d(y[:,:,:,:].ravel(), z1[:,:,:,:].ravel())

    loss_ncc = self.sess.run(self.loss_ncc, feed_dict={self.x: x, self.y: y})
    #loss_NMI = self.sess.run(self.loss_NMI, feed_dict={self.x: x, self.y: y})
    loss_ssim = self.sess.run(self.loss_ssim, feed_dict={self.x: x, self.y: y})
    loss_mse = self.sess.run(self.loss_mse, feed_dict={self.x: x, self.y: y})

    for i in range(z1.shape[0]):
      save_image_with_scale(dir_path+"/{}_x.tif".format(name1[i]), x[i,:,:,0])
      save_image_with_scale(dir_path+"/{}_y.tif".format(name1[i]), y[i,:,:,0])
      save_image_with_scale(dir_path+"/{}_z1.tif".format(name1[i]), z1[i,:,:,:])
      
      #np.savetxt('dvf1_x_{}.txt'.format(name1[i]), v1[i,:,:,0])
      #np.savetxt('dvf1_y_{}.txt'.format(name1[i]), v1[i,:,:,1])

    return loss, loss_ncc, loss_ssim, loss_mse, loss_nmi

  def save(self, dir_path1, dir_path2, dir_path3):
    self.vCNN1.save(self.sess, dir_path1+"/model.ckpt")
    self.vCNN2.save(self.sess, dir_path2+"/model.ckpt")
    self.vCNN3.save(self.sess, dir_path3+"/model.ckpt")

  def restore(self, dir_path1, dir_path2, dir_path3):
    self.vCNN1.restore(self.sess, dir_path1+"/model.ckpt")
    self.vCNN2.restore(self.sess, dir_path2+"/model.ckpt")
    self.vCNN3.restore(self.sess, dir_path3+"/model.ckpt")


