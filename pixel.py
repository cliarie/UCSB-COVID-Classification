import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
 
# get the path/directory
COVID_dir = "/home/claire/data/segmented/COVID_seg/5M9RRKD5/5M9RRKD5_seg.nii.gz"
PNA_dir = "/home/claire/data/segmented/PNA_seg/2AE5UQA7/2AE5UQA7_seg.nii.gz"

pixels = []
img = nib.load(COVID_dir)
for i in range(img.shape[2]):
    pixels.append(np.sum(img.get_fdata()[:,:,i,0]))

print(pixels)

x = np.array(pixels)
print(x)

plt.hist(x, bins=len(x), rwidth=0.5)
plt.ylabel('Amount of Slices')

plt.xlabel('Voxels')

# plt.title('PNA CT Scan')
plt.title('COVID-19 CT Scan')

# plt.savefig("PNA-Voxels.png")
plt.savefig("COVID-Voxels.png")