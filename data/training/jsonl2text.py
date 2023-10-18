import json
import sys

f = open(sys.argv[1])
line = f.readline()

while line:
    line = json.loads(line)
    line = line['text']
    print(line+'\n')
    line = f.readline()
    
