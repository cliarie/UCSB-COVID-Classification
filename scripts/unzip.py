import tarfile, glob
base_dir = '/home/claire/data/seg/COVID/'
    
for name in glob.glob(base_dir + '*.gz'):
     print(name)
     tf = tarfile.open(name)
     tf.extractall(base_dir)
