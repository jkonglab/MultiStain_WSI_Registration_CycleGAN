import tensorflow as tf
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir
import numpy as np

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3" #model will be trained on GPU 0


def main():
  sess = tf.Session()
  config = get_config(is_train=True)
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)

  reg = DIRNet(sess, config, "DIRNet", is_train=True)
  dh = MNISTDataHandler(is_train=True)

  for i in range(config.iteration):
    batch_x, batch_y,name1 = dh.sample_pair(config.batch_size)
    #print(np.shape(batch_x))
    #print(np.shape(batch_y))

    loss = reg.fit(batch_x, batch_y)
    print("iter {:>6d} : {}".format(i+1, loss))

    if (i+1) % 1000 == 0:
      loss_ncc, loss_ssim, loss_mse, loss_nmi = reg.deploy(config.tmp_dir, batch_x, batch_y, name1)
      reg.save(config.ckpt_dir)
      print("iter {:>6d}  ncc loss:{},  ssim loss:{}, mse loss:{}, nmi loss:{}".format(i+1, loss_ncc, loss_ssim, loss_mse, loss_nmi))

if __name__ == "__main__":
  main()
