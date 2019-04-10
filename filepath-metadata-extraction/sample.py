import pandas as pd
import pyperclip as pc
import sys

data = pd.read_csv("SHUP 2D Files Training Data.csv", dtype=str)

def sample(*fields):
    s = data.sample(1)
    v = [s[f].values[0] for f in fields]
    o = ' '.join(v)
    o = o.replace('\\','\\\\')
    pc.copy(o)
    return o

if __name__ == "__main__":
     
    fields = sys.argv[1:]
    query = ''
    while query != 'q':
        print(sample(*fields))
        query = input()