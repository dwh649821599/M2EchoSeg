import glob
import random
import os
import numpy as np
import pickle
import cv2

def split_dataset():
    p_paths = glob.glob('HCM/Philips/HCM*')
    print(len(p_paths))
    train_val_paths = p_paths
    test_paths = p_paths
    return train_val_paths, test_paths

def make_dataset(paths):
    dataset = {}
    frames_num = 0
    for i in paths:
        views = {'A2C1': [], 'A3C1': [], 'A4C1': []}
        for tp in views.keys():
            path = os.path.join(i, 'h', tp)
            image_path = glob.glob(os.path.join(path, 'image*.png'))
            image_path = sorted(image_path, key=lambda x: int(x.split('.png')[0].split('image')[-1]))
            # print(len(image_path))
            if len(image_path) != 0:
                for j in image_path:
                    #label_path = j.replace('image', 'label')
                    #label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    #cv2.imwrite(label_path, label)
                    views[tp].append(j)
                    frames_num += 1
        dataset[i] = views
    print('frames_num:', frames_num)
    return dataset
    

if __name__ == '__main__':
    random.seed(0)
    train_val_paths, test_paths = split_dataset()
    train_val_data = make_dataset(train_val_paths)
    test_data = make_dataset(test_paths)
    print(len(train_val_data), len(test_data))
    pickle.dump(train_val_data, open('HCM/train_val_dataset.pkl', 'wb'))
    pickle.dump(test_data, open('HCM/test_dataset.pkl', 'wb'))
