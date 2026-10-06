import csv
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent / 'data'

def load_csv(path):
    X, y =[], []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        
        for row in reader:
            vals = [float(val) for val in row]
            X.append(vals[:-1])
            y.append(vals[-1])
            
    return X, y



def load_moon():
    return load_csv(f'{DATA_DIR}/moons.csv')



def  load_xor():
    return load_csv(f'{DATA_DIR}/xor.csv') 



def train_test_split(X, y, test_ratio = 0.25, random_seed = 42):
    idx = list(range(len(X)))
    random.Random(random_seed).shuffle(idx)
    
    n_test = int(len(X)*test_ratio)
    train, test = idx[n_test:], idx[:n_test]
    
    return ([X[i] for i in train], [y[i] for i in train],
            [X[i] for i in test], [y[i] for i in test])