import matplotlib.pyplot as plt
import numpy as np
import os

import glob
 
# get the path/directory
COVID_dir = "/home/claire/data/processedsplit/COVID"
PNA_dir = "/home/claire/data/processedsplit/PNA"

slices = []
for scan in glob.iglob(f'{PNA_dir}/*/'):
    #print(scan)
    slices.append(len(os.listdir(scan)))

print(slices)

x = np.array(slices)
# q25, q75 = np.percentile(x, [25, 75])
# bin_width = 2 * (q75 - q25) * len(x) ** (-1/3)
# bins = round((x.max() - x.min()) / bin_width)

# print("Freedman–Diaconis number of bins:", bins)

plt.hist(x, bins=len(x), rwidth=0.5)

plt.ylabel('Number of PNA Scans')
# plt.ylabel('Number of COVID-19 Scans')

plt.xlabel('Number of Slices with Lesions')

plt.title('Typical Pneumonia Lesions')
# plt.title('COVID-19 Lesions') 

plt.savefig("PNA-lesions.png")
# plt.savefig("COVID-lesions.png")
