#!/Users/leeqingming/Downloads/NLPFinalProj/mychinesemultipa/venv/bin/python3.11

import unicodedata
import sys


def main(fn):
    with open(fn, encoding='utf-8') as f:
        print(unicodedata.normalize('NFD', f.read()))


if __name__ == '__main__':
    main(sys.argv[1])
