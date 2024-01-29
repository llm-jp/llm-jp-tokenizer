import argparse
from sklearn.metrics import classification_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gold', help='gold segmentation')
    parser.add_argument('-p', '--pred', help='predicted segmentation')
    args = parser.parse_args()

    def convert2BI(line):
        line = line.split()
        biLine = ''
        for w in line:
            biLine += 'B'
            biLine += 'I'*(len(w)-1)
        return biLine

    gold = [tag for line in open(args.gold) for tag in convert2BI(line.strip())]
    pred = [tag for line in open(args.pred) for tag in convert2BI(line.strip())]

    print(classification_report(gold, pred, digits=3))

if __name__ == '__main__':
    main()
