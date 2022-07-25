f="/home/claire/3ResNet152/analyze/f.txt"
fCP="/home/claire/3ResNet152/analyze/fCP.txt"
fCN="/home/claire/3ResNet152/analyze/fCN.txt"
fNP="/home/claire/3ResNet152/analyze/fNP.txt"

def sorting(filename):
  infile = open(filename)
  words = []
  for line in infile:
    words.append(line)
  infile.close()
  words.sort(key = lambda x: x.split()[1])
  for i in words:
    print(i)

  outfile = open(filename, "w")
  for i in words:
    outfile.writelines(i)
    # outfile.writelines("\n")
  outfile.close()
sorting(f)
sorting(fCP)
sorting(fCN)
sorting(fNP)