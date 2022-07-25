f="/home/claire/3ResNet152/analyze/f copy.txt"
fCP="/home/claire/3ResNet152/analyze/fCP copy.txt"
fCN="/home/claire/3ResNet152/analyze/fCN copy.txt"
fNP="/home/claire/3ResNet152/analyze/fNP copy.txt"
VOX = "/home/claire/data/voxels/voxels.txt"

def analyze(filename):
  infile = open(filename)
  words = []
  for line in infile:
    words.append(line)
  infile.close()
  outfile = open(filename, "w")
  voxfile = open(VOX, "r").readlines()
  for i in words:
    #if i.split(' ')[0] == 'X':
    if 1==1:
        scan = i.split(' ')[1]
        name = scan.split('_slice_')[0]
        slice = scan.split('_slice_')[1]
        slice = slice.split('.')[0]
        str = name + " " + slice
        found = False
        for line in voxfile:
            # search string
            #print(name + " " + slice, line)
            if str in line:
                print('string found in a file', str)
                print('Line:', line)
                pixel = line.split(' ')[2]
                outfile.writelines(pixel + " " + i)
                found = True
                break
        if not found:
            ok = 1
            print(str, type(line))
            outfile.writelines("0.0 " + i)
    else:
        ok=1
        outfile.writelines(i)
  outfile.close()
analyze(f)
analyze(fCP)
analyze(fCN)
analyze(fNP)
