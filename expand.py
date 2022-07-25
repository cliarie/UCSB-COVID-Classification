import os
import glob
import shutil
 
# get the path/directory
COVID_dir = "/home/claire/data/Processed/COVID"
PNA_dir = "/home/claire/data/Processed/PNA"
Normal_dir = "/home/claire/data/Processed/Normal"
COVID_seg = '/home/claire/data/segmented/COVID_seg'
PNA_seg = '/home/claire/data/segmented/PNA_seg'

COVID_output = "/home/claire/data/processed/COVID/"
PNA_output = "/home/claire/data/processed/PNA/"
Normal_output = "/home/claire/data/processed/Normal/"
Cseg_output = '/home/claire/data/seg/COVID'
Pseg_output = '/home/claire/data/seg/PNA'

for image in glob.iglob(f'{PNA_seg}/*/*'):
    print(image)
    shutil.copy2(image, Pseg_output)