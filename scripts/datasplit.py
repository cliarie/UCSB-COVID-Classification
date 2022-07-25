import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import glob
 
# get the path/directory
path = "/home/claire/data/data-split"
classes = ["COVID", "PNA", "Normal"]

# COVID, PNA, Normal
# train = [66, 48, 48]
# val = [17, 13, 12]
# test = [21, 15, 14]

covid = [66,17,21]
pna = [48, 13,15]
normal = [48,12,14]
# data = [train, val, test]
data=[covid,pna,normal]

x = np.array(data)
#print(x)

barWidth = 0.20

br1 = np.arange(3)
br2 = [x + barWidth + 0.05 for x in br1]
br3 = [x + barWidth + 0.05 for x in br2]

plt.bar(br1, data[0], label = "train", color = 'orangered', width = barWidth)
plt.bar(br2, data[1], label = "val", color = 'slateblue', width = barWidth)
plt.bar(br3, data[2], label = "test", color = 'royalblue', width = barWidth)
plt.xticks([r + barWidth for r in range(3)],
        ['train', 'val', 'test'])
plt.legend(labels=classes)
plt.ylabel("CT Scans")
plt.xlabel("Data Splits")
plt.title("CT Scan Dataset Train Val Test")

plt.savefig("Datasplit.png")
