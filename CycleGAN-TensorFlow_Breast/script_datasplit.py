import os
import glob
import shutil
from shutil import copyfile

HE_files = glob.glob("data/HE2Ki67/HE/*.jpg")
Ki_files = glob.glob("data/HE2Ki67/Ki67/*.jpg")

for i in range(6700):
	name1 = os.path.basename(HE_files[i])
	dest_1 = 'data/HE2Ki67/testA/' + name1
	copyfile(HE_files[i], dest_1)

for i in range(6700, len(HE_files)):
	name2 = os.path.basename(HE_files[i])
	dest_2 = 'data/HE2Ki67/trainA/' + name2
	copyfile(HE_files[i], dest_2)


for i in range(6600):
	name3 = os.path.basename(Ki_files[i])
	dest_3 = 'data/HE2Ki67/testB/' + name3
	copyfile(Ki_files[i], dest_3)

for i in range(6600, len(Ki_files)):
	name4 = os.path.basename(Ki_files[i])
	dest_4 = 'data/HE2Ki67/trainB/' + name4
	copyfile(Ki_files[i], dest_4)




