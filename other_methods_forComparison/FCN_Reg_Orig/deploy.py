import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4" #model will be trained on GPU 0


def main():
  sess = tf.Session()
  config = get_config(is_train=False)
  mkdir(config.result_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=False)
  reg.restore(config.ckpt_dir1)
  dh = MNISTDataHandler(is_train=False)
  

  end_idx = 0
  _loss = []
  _loss_ncc = []
  _loss_NMI = []
  _loss_ssim = []
  _loss_mse = []
  counter = 0 

  for i in range(2280):  #7632):  #2:  #1000):
    result_i_dir = config.result_dir+"/{}".format(i)
    mkdir(result_i_dir)

    if i == 0:
	    start_idx = 0
    else:
	    start_idx = end_idx + 1

    end_idx = (start_idx + 1) -1    # 5 is batch_size

    batch_x, batch_y, name1 = dh.sample_pair(config.batch_size, i, start_idx, end_idx)  #config.batch_size
    loss, loss_ncc,  loss_ssim, loss_mse, loss_nmi = reg.deploy(result_i_dir, batch_x, batch_y, name1)
    _loss.append(loss)
    _loss_ncc.append(loss_ncc)
    _loss_NMI.append(loss_nmi)
    _loss_ssim.append(loss_ssim)
    _loss_mse.append(loss_mse)

    counter = counter + config.batch_size
    print("folder number,{:>6d} test loss:{}, ncc loss:{},  ssim loss:{}, mse loss:{}, nmi loss:{}".format(i, loss, loss_ncc,  loss_ssim, loss_mse, loss_nmi))    
  """
  batch_x, batch_y, name1 = dh.sample_pair(config.batch_size)
  reg.deploy(config.result_dir, batch_x, batch_y, name1)
  """

  print('counter:{}, total avg total_loss:{}, ncc loss:{}, ssim loss:{}, mse loss: {}, nmi loss:{}'.format(counter, np.mean(_loss), np.mean(_loss_ncc),  np.mean(_loss_ssim), np.mean(_loss_mse), np.mean(_loss_NMI)))

if __name__ == "__main__":
  main()

