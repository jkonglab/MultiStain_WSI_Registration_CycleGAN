class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.batch_size = 5 #4  #10 #64
    config.im_size = [256, 256]    #[512, 512]   #[1024, 1024]  #[256, 256]    #[28, 28]
    config.lr =  1e-6  #1e-6   #1e-4
    config.iteration = 15000 #10000
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt" # for vCNN
    #config.ckpt_dir2 = "ckpt2" # for aCNN
    #config.epoch_num = 10  #500
    #config.save_interval = 2

  else:
    config.batch_size = 1 #5 #10 #221  #10
    config.im_size =  [256, 256]    #[512, 512]  #[1024, 1024]  #[256, 256]   #[28, 28]
    config.result_dir = "result"
    config.ckpt_dir1 = "ckpt"
    #config.ckpt_dir2 = "ckpt2"
  return config
