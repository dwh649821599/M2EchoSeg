import glob
import random
import os
import numpy as np
import pickle
import nibabel as nib
import cv2

def video2image():
    paths = glob.glob('CAMUS_public/database_nifti/patient*')
    for path in paths:
        A2C_image_path = glob.glob(os.path.join(path, 'patient*_2CH_half_sequence.nii.gz'))[0]
        A2C_label_path = glob.glob(os.path.join(path, 'patient*_2CH_half_sequence_gt.nii.gz'))[0]
        A4C_image_path = glob.glob(os.path.join(path, 'patient*_4CH_half_sequence.nii.gz'))[0]
        A4C_label_path = glob.glob(os.path.join(path, 'patient*_4CH_half_sequence_gt.nii.gz'))[0]
        A2C_image = nib.load(A2C_image_path)
        A2C_label = nib.load(A2C_label_path)
        A4C_image = nib.load(A4C_image_path)
        A4C_label = nib.load(A4C_label_path)
        save_path = os.path.join('CAMUS', path.split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(A2C_image.shape[2]):
            image = A2C_image.get_fdata()[:, :, i]
            label = A2C_label.get_fdata()[:, :, i]
            image = image.astype(np.uint8)
            label = label.astype(np.uint8)
            label[label != 2] = 0
            label[label == 2] = 255
            A2C_save_path = os.path.join(save_path, 'A2C')
            if not os.path.exists(A2C_save_path):
                os.makedirs(A2C_save_path)
            cv2.imwrite(os.path.join(A2C_save_path, f'image{i}.png'), image.T)
            cv2.imwrite(os.path.join(A2C_save_path, f'label{i}.png'), label.T)
        for i in range(A4C_image.shape[2]):
            image = A4C_image.get_fdata()[:, :, i]
            label = A4C_label.get_fdata()[:, :, i]
            image = image.astype(np.uint8)
            label = label.astype(np.uint8)
            label[label != 2] = 0
            label[label == 2] = 255
            A4C_save_path = os.path.join(save_path, 'A4C')
            if not os.path.exists(A4C_save_path):
                os.makedirs(A4C_save_path)
            cv2.imwrite(os.path.join(A4C_save_path, f'image{i}.png'), image.T)
            cv2.imwrite(os.path.join(A4C_save_path, f'label{i}.png'), label.T)

def split_dataset():
    paths = glob.glob('CAMUS/patient*')
    print(len(paths))
    random.shuffle(paths)
    train_val_paths = paths[:int(len(paths)*0.9)]
    test_paths = paths[int(len(paths)*0.9):]
    print(len(train_val_paths), len(test_paths))
    return train_val_paths, test_paths

def make_dataset(paths):
    dataset = {}
    frames_num = 0
    for i in paths:
        views = {'A2C': [], 'A4C': []}
        for tp in views.keys():
            path = os.path.join(i, tp)
            image_path = glob.glob(os.path.join(path, 'image*.png'))
            image_path = sorted(image_path, key=lambda x: int(x.split('.png')[0].split('image')[-1]))
            if len(image_path) != 0:
                for j in image_path:
                    views[tp].append(j)
                    frames_num += 1
        dataset[i] = views
    print('frames_num:', frames_num)
    return dataset
    

if __name__ == '__main__':
    random.seed(0)
    video2image()
    train_val_paths, test_paths = split_dataset()
    train_val_data = make_dataset(train_val_paths)
    test_data = make_dataset(test_paths)
    print(len(train_val_data))
    print(len(test_data))
    pickle.dump(train_val_data, open('CAMUS/train_val_dataset.pkl', 'wb'))
    pickle.dump(test_data, open('CAMUS/test_dataset.pkl', 'wb'))