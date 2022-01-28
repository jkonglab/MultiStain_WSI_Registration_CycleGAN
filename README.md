# Multi-modal-Registration

## Stage1:

## CycleGAN-Tensorflow-Breast for image translation:

# Environment
TensorFlow 1.0.0
Python 3.6.0

First move to the desired folder
```
$ cd CycleGAN-TensorFlow_Breast 
```

# Data preparing

First download the dataset e.g. HE2Ki67

```
$ bash download_dataset.sh HE2Ki67
```

Write the dataset to tfrecords
```
$ python build_data.py
```
Check $ python3 build_data.py --help for more details.

# Training
```
$ python train.py
```

To change some default settings, pass those to the command line, such as:
```
$ python train.py  
    --X=data/tfrecords/HE.tfrecords 
    --Y=data/tfrecords/Ki67.tfrecords
```

Here is the list of arguments:

```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--image_size IMAGE_SIZE]
                [--use_lsgan [USE_LSGAN]] [--nouse_lsgan]
                [--norm NORM] [--lambda1 LAMBDA1] [--lambda2 LAMBDA2]
                [--learning_rate LEARNING_RATE] [--beta1 BETA1]
                [--pool_size POOL_SIZE] [--ngf NGF] [--X X] [--Y Y]
                [--load_model LOAD_MODEL]
```

optional arguments:
```
  -h, --help             show this help message and exit 
  --batch_size BATCH_SIZE
                        batch size, default: 5 
  --image_size IMAGE_SIZE
                        image size, default: 256 
  --use_lsgan [USE_LSGAN]
                        use lsgan (mean squared error) or cross entropy loss,
                        default: True 
  --nouse_lsgan         
  --norm NORM           [instance, batch] use instance norm or batch norm,
                        default: instance 
  --lambda1 LAMBDA1     weight for forward cycle loss (X->Y->X), default: 10.0 
  --lambda2 LAMBDA2     weight for backward cycle loss (Y->X->Y), default:
                        10.0
  --learning_rate LEARNING_RATE
                        initial learning rate for Adam, default: 0.0002
  --beta1 BETA1         momentum term of Adam, default: 0.5
  --pool_size POOL_SIZE
                        size of image buffer that stores previously generated
                        images, default: 50
  --ngf NGF             number of gen filters in first conv layer, default: 64
  --X X                 X tfrecords file for training, default:
                        data/tfrecords/apple.tfrecords
  --Y Y                 Y tfrecords file for training, default:
                        data/tfrecords/orange.tfrecords
  --load_model LOAD_MODEL
                        folder of saved model that you wish to continue
                        training (e.g. 20170602-1936), default: None
```

If the training process is halted and want to continue training, then the load_model parameter can be set like this.
```
$ python train.py  
    --load_model 20170602-1936
```
## Notes:
- If high constrast background colors between input and generated images are observed (e.g. black becomes white), you should restart your training.
- Train several times to get the best models.

# Export model
Export from a checkpoint to a standalone GraphDef file as follow:

```
$ python export_graph.py --checkpoint_dir checkpoints/${datetime} \
                          --XtoY_model HE2Ki67.pb \
                          --YtoX_model Ki672HE.pb \
                          --image_size 256
```
# Inference
After exporting model, it can be used for inference. For example:
```
python inference.py --model pretrained/HE2Ki67.pb \
                     --input input_sample.jpg \
                     --output output_sample.jpg \
                     --image_size 256
```

## Stage2:

## Multi-scale FCN for image registration:

Move to the desired folderFCN_Reg_Orig
```
$ cd MultiscaleFCN_Reg_synMonomodal
```

Tensorflow implementation of Multi-scale FCN

```
# Training
python train.py
```
Intermediate results and model checkpoint can be found in tmp and ckpt

```
# Inference
python deploy.py
```
Evaluation results can be found in result folder.

## Comparison with other models

There are some standard registration models like DirNet, FCN which are compared for three different datasets which are multi-1, synmono-1 and synmono-2 respectively. For each model there are three different folders associated with each dataset.

- DIRNet-tensorflow_MultiModal : DirNet model trained with multi-1 dataset
- DIRNet-tensorflow_MonoModal_origCycleGAN : DirNet model trained with synmono-1 dataset
- DIRNet-tensorflow_MonoModal_modCycleGAN : DirNet model trained with synmono-2 dataset
- FCN_Reg_Orig: FCN model trained with multi-1 dataset
- FCN_Reg_Monomodal_origCycleGAN : FCN model trained with synmono-1 dataset
- FCN_Reg_Monomodal_modCycleGAN: FCN model trained with synmono-2 dataset

Move to the desired folder
```
cd other_methods_forComparison/DIRNet-tensorflow_MultiModal
```
Tensorflow implementation of DirNet

```
# Training
python train.py
```
Intermediate results and model checkpoint can be found in tmp and ckpt

```
# Inference
python deploy.py
```
Evaluation results can be found in result folder.

Similar for other comparison models.


