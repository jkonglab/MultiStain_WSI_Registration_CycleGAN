import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir
import numpy as np


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4" #model will be trained on GPU 0



def main():
  sess = tf.Session()
  config = get_config(is_train=False)
  mkdir(config.result_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=False)
  reg.restore(config.ckpt_dir)
  dh = MNISTDataHandler(is_train=False)
  

  end_idx = 0
  _loss_ncc = []
  _loss_NMI = []
  _loss_ssim = []
  _loss_mse = []

  for i in range(2280):
    result_i_dir = config.result_dir+"/{}".format(i)
    mkdir(result_i_dir)

    if i == 0:
	    start_idx = 0
    else:
	    start_idx = end_idx + 1

    end_idx = (start_idx + 1) -1   # batch size = 1

    batch_x, batch_y, name1 = dh.sample_pair(config.batch_size, i, start_idx, end_idx)
    loss_ncc,  loss_ssim, loss_mse, loss_nmi = reg.deploy(result_i_dir, batch_x, batch_y, name1)
    _loss_ncc.append(loss_ncc)
    _loss_ssim.append(loss_ssim)
    _loss_mse.append(loss_mse)
    _loss_NMI.append(loss_nmi)

    print("folder number,{:>6d} ncc loss:{},  ssim loss:{}, mse loss:{}, nmi loss:{}".format(i, loss_ncc,  loss_ssim, loss_mse, loss_nmi))

  """
  batch_x, batch_y, name1 = dh.sample_pair(config.batch_size)
  reg.deploy(config.result_dir, batch_x, batch_y, name1)
  """

  print('total avg  ncc loss:{}, ssim loss:{}, mse loss: {}, nmi loss:{}'.format(np.mean(_loss_ncc),  np.mean(_loss_ssim), np.mean(_loss_mse), np.mean(_loss_NMI)))

if __name__ == "__main__":
  main()


