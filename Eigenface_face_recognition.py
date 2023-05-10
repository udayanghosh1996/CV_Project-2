
import numpy as np
import cv2
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split





def load_images(path, lst):
    imgs = []
    for item in lst:
        imgs.append(cv2.resize(cv2.imread(os.path.join(path, item),0), (256, 256)))
    return np.array(imgs)





def img_show(img):
    plt.imshow(img, cmap = 'gray')





def get_flatten_vector(images):
    flat = []
    for image in images:
        flat.append(image.flatten())
    return np.array(flat)





def get_flatten_vector_test(image):
    return image.flatten()





def get_eigen_faces(flat_images):
    pca = PCA().fit(flat_images)
    egn_fc = pca.components_[:60]
    return egn_fc, pca





def get_weights(egn_fc, flat_images, pca):
    weights = np.matmul(egn_fc, (flat_images-pca.mean_).T)
    return weights





def get_euclidean_dist(sample_weight, test_weight):
    dist = np.linalg.norm(sample_weight - test_weight, axis=0)
    return dist



def get_topk_match(sample_weights, test_weights, k):
    dist = np.argsort(get_euclidean_dist(sample_weights, test_weights))
    return dist[:k]





def plot_top_k_image(flatten_img, top_k):
    fig = plt.figure(figsize=(30, 30))
    for idx, k in enumerate(top_k):
        fig.add_subplot(int(len(top_k)/2)+1, 2, idx+1)
        plt.imshow(flatten_img[k].reshape(256, 256), cmap = 'gray')
        plt.title('Top-'+str(idx+1))
    plt.show()




def load_label(path):
    df = pd.read_csv(path)
    return df["Name"]




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
    labels = load_label(os.path.join(os.getcwd(), 'Label.csv'))
    train_imgs, test_imgs, train_label, test_label = train_test_split(imgs, labels, test_size=0.2,
                                                                      random_state=60, shuffle=True)
    flatten_train_images = get_flatten_vector(train_imgs)
    train_egn_fc, train_pca = get_eigen_faces(flatten_train_images)
    train_weights = get_weights(train_egn_fc, flatten_train_images, train_pca)
    dect_face = []
    for images in test_imgs:
        flatten_test_images = get_flatten_vector(np.array([images]))
        test_egn_fc, test_pca = get_eigen_faces(flatten_test_images)
        test_weights = get_weights(test_egn_fc, flatten_test_images, test_pca)
        top_k = get_topk_match(train_weights, test_weights, 2)
        dect_face.append(train_label[top_k[1]])
    acc = get_accuracy(dect_face, test_label)
    print('Face detection accuracy is: ', acc)






