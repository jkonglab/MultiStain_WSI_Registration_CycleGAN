import tensorflow as tf
from WarpST import WarpST
from ops import *
from bicubic_interp import bicubic_interp_2d

class CNN(object):
  def __init__(self, name, is_train):
    self.name = name
    self.is_train = is_train
    self.reuse = None
  
  def __call__(self, x):
    with tf.variable_scope(self.name, reuse=self.reuse):
      x = conv2d(x, "conv1", 64, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")

      x = conv2d(x, "conv2", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
      x = conv2d(x, "out1", 128, 3, 1, "SAME", True, tf.nn.elu, self.is_train)
      x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "SAME")
      
      x = conv2d(x, "out2", 2, 3, 1, "SAME", False, None, self.is_train)

    if self.reuse is None:
      self.var_list = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
      self.saver = tf.train.Saver(self.var_list)
      self.reuse = True

    return x

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
    self.x = tf.placeholder(tf.float32, im_shape)
    self.y = tf.placeholder(tf.float32, im_shape)
    self.xy = tf.concat([self.x, self.y], 3)

    self.vCNN = CNN("vector_CNN", is_train=self.is_train)

    # vector map & moved image
    self.v = self.vCNN(self.xy)
    self.z, self.im_flat, self.Ia, self.Ib, self.Ic, self.Id = WarpST(self.x, self.v, config.im_size)

    self.V = bicubic_interp_2d(self.v, config.im_size)

    self.loss_ncc = -ncc(self.y, self.z)
    loss_DSSIM = DSSIMObjective()
    self.loss_ssim = loss_DSSIM(self.y, self.z)
    self.loss_mse = mse(self.y, self.z)
    
    if self.is_train :
      self.loss = ncc(self.y, self.z)
      #self.loss = mse(self.y, self.z)


      self.optim = tf.train.AdamOptimizer(config.lr)
      self.train = self.optim.minimize(
        - self.loss, var_list=self.vCNN.var_list)

    #self.sess.run(
    #  tf.variables_initializer(self.vCNN.var_list))
    self.sess.run(
      tf.global_variables_initializer())

  def fit(self, batch_x, batch_y):
    _, loss = \
      self.sess.run([self.train, self.loss], 
      {self.x:batch_x, self.y:batch_y})
    return loss

  def deploy(self, dir_path, x, y, name1):
    z,v, V, im_flat, Ia, Ib, Ic, Id = self.sess.run([self.z, self.v, self.V, self.im_flat, self.Ia, self.Ib, self.Ic, self.Id], {self.x:x, self.y:y})

    channels = tf.shape(z)[3]
    z_flat = tf.reshape(z, tf.stack([-1, channels]))
    z_flat = tf.cast(z_flat, 'float32')
    z_flat = self.sess.run(z_flat, {self.x:x, self.y:y})

    loss_nmi = mutual_information_2d(y[:,:,:,:].ravel(), z[:,:,:,:].ravel())

    loss_ncc = self.sess.run(self.loss_ncc, feed_dict={self.x: x, self.y: y})
    loss_ssim = self.sess.run(self.loss_ssim, feed_dict={self.x: x, self.y: y})
    loss_mse = self.sess.run(self.loss_mse, feed_dict={self.x: x, self.y: y})

    #print("final warped image:{}".format(z))
    #print("dvf:{}".format(v))
    #print("final warped image flat:{}".format(z_flat[5000:6000]))
    #print("input image flat:{}".format(im_flat[5000:6000]))
    #print("Ia:{}".format(Ia[:30]))
    #print("Ib:{}".format(Ib[:30]))
    #print("Ic:{}".format(Ic[:30]))
    #print("Id:{}".format(Id[:30]))

    for i in range(z.shape[0]):
      #save_image_with_scale(dir_path+"/{:02d}_x.tif".format(i+1), x[i,:,:,0])
      #save_image_with_scale(dir_path+"/{:02d}_y.tif".format(i+1), y[i,:,:,0])
      #save_image_with_scale(dir_path+"/{:02d}_z.tif".format(i+1), z[i,:,:,0])
      save_image_with_scale(dir_path+"/{}_x.tif".format(name1[i]), x[i,:,:,0])
      save_image_with_scale(dir_path+"/{}_y.tif".format(name1[i]), y[i,:,:,0])
      save_image_with_scale(dir_path+"/{}_z.tif".format(name1[i]), z[i,:,:,:])
      #save_image_with_scale(dir_path+"/{}_v1.tif".format(name1[i]), v[i,:,:,0])
      #save_image_with_scale(dir_path+"/{}_v2.tif".format(name1[i]), v[i,:,:,1])
      #np.savetxt('Dirnet_dvf1_x_{}.txt'.format(name1[i]), V[i,:,:,0])
      #np.savetxt('Dirnet_dvf1_y_{}.txt'.format(name1[i]), V[i,:,:,1])

    return loss_ncc, loss_ssim, loss_mse, loss_nmi

  def save(self, dir_path):
    self.vCNN.save(self.sess, dir_path+"/model.ckpt")

  def restore(self, dir_path):
    self.vCNN.restore(self.sess, dir_path+"/model.ckpt")


