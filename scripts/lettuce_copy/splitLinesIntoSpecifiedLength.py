import sys

maxLength = 1000

f = open(sys.argv[1], errors='ignore')
line = f.readline()

while line:
    line = line.rstrip()
    length = len(line)
    if length < maxLength:
        print(line + '\n')
    else:
        for i in range(0, length, maxLength):
            print(line[i:i+maxLength] + '\n')
    line = f.readline()

