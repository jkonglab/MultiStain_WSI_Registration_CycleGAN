import os
import glob
import tensorflow as tf
from scipy import misc
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2" #model will be trained on GPU 0


#path1 = 'Fixed/'
#path2 = 'Fixed_set2/'

#file_list = os.listdir(path1)
#for j in range(12000):
#	source = os.path.join(path1 + file_list[j])
#	destination = os.path.join(path2 +  file_list[j])
#	os.rename(source, destination)


path = '../rigid_CASE9_HES_56319_31672/'
folder_list = os.listdir(path)
#file_list = os.listdir(path + 'rigid_CASE50_HES_21426_37427/')

path2 = 'result_rigid_CASE9_HES_56319_31672/'

#cmd1 = 'python export_graph.py --checkpoint_dir checkpoints/20210313-1259  --XtoY_model HE2Ki67.pb --YtoX_model Ki672HE.pb --image_size 256'
#os.system(cmd1)


#for i in range(100):  #len(folder_list)):
file_list = os.listdir(path)
for j in range(923):  #len(file_list)):
	input_image_name = os.path.join(path + file_list[j])
	output_image_name = os.path.join(path2 +  file_list[j])
	if not os.path.exists(output_image_name):
		cmd2 = 'python inference.py --model pretrained/HE2Ki67.pb --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)
		os.system(cmd2)


"""

## Estimate some quantitative metrics ###
# For quantitative metric use 1000 samples for prediction, and based on that compute the values 
## SSIM, PSNR, FID (Frechet inception distance) 

#HE -> synKi67 -> synHE 

path = 'data/HE2Ki67/testA/'
folder_list = os.listdir(path)

path2 = 'result_synKi67fromHE_Images/'
if not os.path.exists(path2):
	os.makedirs(path2)


file_list = os.listdir(path)
#for j in range(1000):
#	input_image_name = os.path.join(path + file_list[j])
#	output_image_name = os.path.join(path2 +  file_list[j])
#	cmd2 = 'python inference.py --model pretrained/HE2Ki67.pb --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)
#	os.system(cmd2)

file_list2 = os.listdir(path2)

path3 = 'result_synHEfromsynKi67_Images/'
if not os.path.exists(path3):
	os.makedirs(path3)


#for ifile in file_list2:
#	input_image_name = os.path.join(path2 + ifile)
#	output_image_name = os.path.join(path3 + ifile)
#	cmd3 = 'python inference.py --model pretrained/Ki672HE.pb --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)
#	os.system(cmd3)

#Compare values for each image case then take the average of all #

sum_mse = 0
sum_ssim1 = 0
sum_psnr1 = 0
sum_psnr2 = 0

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


for ifile in file_list2:
	#img1 = tf.io.decode_jpeg(os.path.join(path + ifile))  #original HE 
	#img2 = tf.io.decode_jpeg(os.path.join(path3 + ifile))  # syn HE
	img1 = misc.imread(os.path.join(path + ifile))
	img2 = misc.imread(os.path.join(path3 + ifile))
	m = mse(img1, img2) 
	
	img1_tf = tf.Variable(img1)
	img2_tf = tf.Variable(img2)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	
	ssim1_var = tf.image.ssim(img1_tf, img2_tf, max_val=255)
	ssim1 = sess.run(ssim1_var)

	psnr1_var = tf.image.psnr(img1_tf, img2_tf, max_val=255)
	psnr1 = sess.run(psnr1_var)

	im1 = tf.image.convert_image_dtype(img1_tf, tf.float32)
	im2 = tf.image.convert_image_dtype(img2_tf, tf.float32)

	psnr2_var = tf.image.psnr(im1, im2, max_val=1.0)
	psnr2 = sess.run(psnr2_var) 

	sum_mse += m
	sum_ssim1 += ssim1
	sum_psnr1 += psnr1
	sum_psnr2 += psnr2
	
	file1 = open("metric_values_casewise_HEimages.txt","a")
	L = ["Image_name:{} \t, mse:{} \t, ssim1:{} \t, psnr1:{} \t, psnr2:{} \n".format(ifile, m, ssim1, psnr1, psnr2)]
	file1.writelines(L)
	file1.close()
	sess.close()

avg_mse = sum_mse / 1000
avg_ssim1 = sum_ssim1 / 1000
avg_psnr1 = sum_psnr1 / 1000
avg_psnr2 = sum_psnr2 / 1000

file1 = open("metric_values_casewise_HEimages.txt","a")

L = ["Modified_1:loss ssim_l1l2_a: \t, mean mse:{} \t,  mean ssim1:{} \t, mean psnr1:{} \t, mean psnr2:{} \n".format(avg_mse, avg_ssim1, avg_psnr1, avg_psnr2)]
file1.writelines(L)
file1.close()



###### Also for the reverse direction ######
# Ki67 -> syn HE -> syn ki67

path = 'data/HE2Ki67/testB/'
folder_list = os.listdir(path)

path2 = 'result_synHEfromKi67_Images/'
if not os.path.exists(path2):
	os.makedirs(path2)


file_list = os.listdir(path)

#for j in range(1000):
#	input_image_name = os.path.join(path + file_list[j])    # Ki67
#	output_image_name = os.path.join(path2 +  file_list[j]) # syn HE
#	cmd2 = 'python inference.py --model pretrained/Ki672HE.pb --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)
#	os.system(cmd2)


file_list2 = os.listdir(path2)

path3 = 'result_synKi67fromsynHE_Images/'
if not os.path.exists(path3):
	os.makedirs(path3)


#for ifile in file_list2:
#	input_image_name = os.path.join(path2 + ifile)   # syn HE
#	output_image_name = os.path.join(path3 + ifile)  # syn Ki67
#	cmd3 = 'python inference.py --model pretrained/HE2Ki67.pb  --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)
#	os.system(cmd3)


#Compare values for each image case then take the average of all #

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


sum_mse = 0
sum_ssim1 = 0
sum_psnr1 = 0
sum_psnr2 = 0

for ifile in file_list2:
	#img1 = tf.decode_jpg(os.path.join(path + ifile))  #original Ki67 
	#img2 = tf.decode_jpg(os.path.join(path3 + ifile))  # syn ki67
	
	img1 = misc.imread(os.path.join(path + ifile))
	img2 = misc.imread(os.path.join(path3 + ifile))
	m = mse(img1, img2)

	img1_tf = tf.Variable(img1)
	img2_tf = tf.Variable(img2)
	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	
	ssim1_var = tf.image.ssim(img1_tf, img2_tf, max_val=255)
	ssim1 = sess.run(ssim1_var)
	
	psnr1_var = tf.image.psnr(img1_tf, img2_tf, max_val=255)
	psnr1 = sess.run(psnr1_var)

	im1 = tf.image.convert_image_dtype(img1_tf, tf.float32)
	im2 = tf.image.convert_image_dtype(img2_tf, tf.float32)
	
	psnr2_var = tf.image.psnr(im1, im2, max_val=1.0)
	psnr2 = sess.run(psnr2_var) 
	
	sum_mse += m
	sum_ssim1 += ssim1
	sum_psnr1 += psnr1
	sum_psnr2 += psnr2
	
	file1 = open("metric_values_casewise_Ki67images.txt","a")
	L = ["Image_name:{} \t, mse:{} \t, ssim1:{} \t, psnr1:{} \t, psnr2:{} \n".format(ifile, m, ssim1, psnr1, psnr2)]
	file1.writelines(L)
	file1.close()


avg_mse = sum_mse / 1000
avg_ssim1 = sum_ssim1 / 1000
avg_psnr1 = sum_psnr1 / 1000
avg_psnr2 = sum_psnr2 / 1000

file1 = open("metric_values_casewise_Ki67images.txt","a")

L = ["Modified_3:perceptual loss vgg19: \t, mean mse:{} \t, mean ssim1:{} \t, mean psnr1:{} \t, mean psnr2:{} \n".format(avg_mse, avg_ssim1, avg_psnr1, avg_psnr2)]
file1.writelines(L)
file1.close()

"""
