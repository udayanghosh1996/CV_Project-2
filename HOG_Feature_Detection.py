
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd




def load_images(path, lst):
    imgs = []
    for item in lst:
        imgs.append(cv2.resize(cv2.imread(os.path.join(path, item),0), (256, 256)))
    return np.array(imgs)




def load_label(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    label = le.fit_transform(df['Name'])
    df = pd.concat([df, pd.DataFrame(label, columns = ['Class'])], axis=1)
    return df[['Name', 'Class']]




def get_hog_image(images):
    hog_ = []
    for idx, image in enumerate(images):
        if idx%1000 == 0:
            #print(idx/1000+1)
        _, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=False)
        hog_.append(hog_image)
    return np.array(hog_)




def resize_image(imgs):
    image = []
    for img in imgs:
        image.append(cv2.resize(img, (150, 150), interpolation = cv2.INTER_LINEAR))
    image = np.array(image)
    print(image.shape)
    return image




def get_accuracy(dect_lbl, true_lbl):
    count = 0
    for dect, true in zip(dect_lbl, true_lbl):
        if dect == true:
            count += 1
    return round(count/len(true_lbl)*100, 2)




if __name__ == "__main__":
    img_path = os.path.join(os.getcwd(), 'All_Images')
    lst = os.listdir(img_path)
    imgs = load_images(img_path, lst)
    lbls = load_label(os.path.join(os.getcwd(), 'Label.csv'))
    labels = lbls['Class']
    train_imgs, test_imgs, train_label, test_label = train_test_split(imgs, labels, test_size=0.2,
                                                                  random_state=60, shuffle=True)
    hog_train = get_hog_image(train_imgs)
    hog_test = get_hog_image(test_imgs)
    clf = SVC()
    clf.fit(hog_train.reshape(hog_train.shape[0], -1), train_label)
    dec_class = clf.predict(hog_test.reshape(hog_test.shape[0], -1))
    acc = get_accuracy(dec_class, test_label['Class'].to_numpy())
    print('Face detection accuracy is: ', acc)

