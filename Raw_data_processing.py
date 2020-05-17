import pandas as pd
import numpy as np
import time
from sklearn import model_selection
from pathlib import Path
from imageio import imread

def process_raw_data():
    t = time.time()
    
    train_folder = Path('../data/imgs/train')

    for classname in train_folder.iterdir():
        X = []
        Y = []
        if classname.name != '.DS_Store':
            for img in classname.iterdir():
                X.append(imread(img))
                Y.append(int(classname.name[1:]))
            X = np.array(X).reshape(-1,3,480,640)
            Y = np.array(Y)
            pd.Series({classname.name: (X,Y)}).to_pickle('./stored_data/' + classname.name + '.pkl')
    print(time.time()-t)


def store_data():
    pass

process_raw_data()
# Xtr, Xte, Ytr, Yte = model_selection.train_test_split(X, Y, test_size=0.25, random_state=20)
# print('Xtr',Xtr.shape)
# print('Ytr',Ytr.shape)
# print('Xte',Xte.shape)
# print('Yte',Yte.shape)
# pd.Series((Xtr,Xte,Ytr,Yte)).to_pickle('data.pkl')


