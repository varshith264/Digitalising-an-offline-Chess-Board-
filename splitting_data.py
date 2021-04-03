
#this file splits entire data set into 2:1 ration for train and test. and stores them in local desktop

import os
import random
import sys
import os
import glob
import shutil

names = ['bb/', 'bk/', 'bn/', 'bp/', 'bq/', 'br/',
         'empty/', 'wb/', 'wk/', 'wn/', 'wp/', 'wq/', 'wr/']
director = '/home/chanduri/Desktop/final_images/data/'

for sub in names:
    directory = director+sub
    sz = 0

    for filename in os.listdir(directory):
        sz += 1

    lis = []

    
    for i in range(sz):
        if(i <= sz//3):
            lis.append(1)
        else:
            lis.append(2)
        i += 1

    random.shuffle(lis)

    i = 0

    for f in os.listdir(directory):
        if(lis[i] == 2):
            shutil.copy(
                directory+f, ('/home/chanduri/Desktop/splitted_data/train/'+sub))
        else:
            shutil.copy(
                directory+f, ('/home/chanduri/Desktop/splitted_data/test/'+sub))
        i += 1
