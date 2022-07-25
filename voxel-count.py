import os
import nibabel as nib
import numpy as np
import glob

COVID_SEG = '/home/claire/data/segmented/COVID_seg'
PNA_SEG = '/home/claire/data/segmented/PNA_seg'

OUT = '/home/claire/data/voxels/voxels.txt'

def get_slices(image, name):
    # slices = []
    outfile = open(OUT, "a")
    for i in range(image.shape[2]):
        pix = np.sum(image.get_fdata()[:,:,i,0])
        outfile.writelines(name.split('/')[6] + " " + str(i) + " " + str(pix) +'\n')
          # outfile.writelines("\n")
    outfile.close()

        # if np.sum(image.get_fdata()[:,:,i,0]) >= 1000:
        #     slices.append(i)
    #print(pixmax)
    # print(max(pixmax))
    # return slices
    # return pixmax

imagelist = glob.glob(COVID_SEG + "/*/*_seg.nii.gz")
for i in imagelist:
  img = nib.load(i)
  get_slices(img, i)
