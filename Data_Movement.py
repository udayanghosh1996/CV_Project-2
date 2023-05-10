
import os
import numpy as np
import pandas as pd
import shutil




SAVE_DIR = os.path.join(os.path.join(os.getcwd(),'All_Images'))
os.makedirs(SAVE_DIR, exist_ok=True)

label = []




for folder in os.listdir(os.path.join(os.getcwd(), 'Images')):
    for file in os.listdir(os.path.join(os.path.join(os.getcwd(), 'Images'), folder)):
        src = os.path.join(os.path.join(os.path.join(os.getcwd(), 'Images'), folder), file)
        shutil.copy(src, SAVE_DIR)
        label.append(folder)


        
label = np.array(label)
df = pd.DataFrame(label, columns=['Name'])
df.to_csv('Label.csv')

