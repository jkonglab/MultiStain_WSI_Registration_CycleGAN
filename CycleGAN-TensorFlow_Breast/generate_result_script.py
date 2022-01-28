import os
import glob


path = '../Reg_Dataset_HE_Patches/pcr/'
file_list = os.listdir(path + 'rigid_CASE50_HES_21426_37427/')

path2 = '../Reg_Dataset_Ki67_syn_Patches/pcr/'

#cmd1 = 'python export_graph.py --checkpoint_dir checkpoints/20210313-1259  --XtoY_model HE2Ki67.pb --YtoX_model Ki672HE.pb --image_size 256'
#os.system(cmd1)


for i in range(len(file_list)):
	input_image_name = os.path.join(path + 'rigid_CASE50_HES_21426_37427/' + file_list[i])
	output_image_name = os.path.join(path2 + 'rigid_CASE50_HES_21426_37427/' + file_list[i])

	cmd2 = 'python inference.py --model pretrained/HE2Ki67.pb --input {} --output {}  --image_size 256'.format(input_image_name, output_image_name)

	os.system(cmd2)


