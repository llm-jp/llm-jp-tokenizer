import sys

f = open(sys.argv[1])
line = f.readline()
i = 0

ids = [int(line.strip()) for line in open(sys.argv[2])][::-1]

while line:
    if i==ids[-1]:
        line = line.rstrip()
        print(line)
        ids.pop()
        if len(ids)==0:
            break
    line = f.readline()
    i += 1    
