from sklearn.model_selection import train_test_split
import os

#COVID
path = '/home/claire/data-split/COVID'

data = os.listdir('/home/claire/data/COVID')
print(len(data))
x_train, x_test = train_test_split(data, test_size=0.2, random_state=15)
x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=25)
print(len(x_train))
print(len(x_val))
print(len(x_test))
file1 = open(os.path.join(path , 'test_COVID.txt'), "w")
for i in x_test:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'train_COVID.txt'), "w")
for i in x_train:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'val_COVID.txt'), "w")
for i in x_val:
    file1.write(str(i) + "\n")
file1.close()

#Normal
path = '/home/claire/data-split/Normal'

data = os.listdir('/home/claire/data/Normal')
print(len(data))
x_train, x_test = train_test_split(data, test_size=0.2, random_state=15)
x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=25)
print(len(x_train))
print(len(x_val))
print(len(x_test))
file1 = open(os.path.join(path , 'test_Normal.txt'), "w")
for i in x_test:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'train_Normal.txt'), "w")
for i in x_train:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'val_Normal.txt'), "w")
for i in x_val:
    file1.write(str(i) + "\n")
file1.close()

#PNA
path = '/home/claire/data-split/PNA'

data = os.listdir('/home/claire/data/PNA')
print(len(data))
x_train, x_test = train_test_split(data, test_size=0.2, random_state=15)
x_train, x_val = train_test_split(x_train, test_size=0.2, random_state=25)
print(len(x_train))
print(len(x_val))
print(len(x_test))
file1 = open(os.path.join(path , 'test_PNA.txt'), "w")
for i in x_test:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'train_PNA.txt'), "w")
for i in x_train:
    file1.write(str(i) + "\n")
file1.close()

file1 = open(os.path.join(path , 'val_PNA.txt'), "w")
for i in x_val:
    file1.write(str(i) + "\n")
file1.close()
