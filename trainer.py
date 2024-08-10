import glob
import math
import os
import shutil
import tempfile
import copy
import time
from typing import Any
import numpy as np
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.fft as fft
from torch.optim import Adam, SGD
import monai
from monai.data import (
    CacheDataset,
    DataLoader,
    ThreadDataLoader,
    Dataset,
    decollate_batch,
    set_track_meta,
    PILReader,
    partition_dataset,
    partition_dataset_classes,
    ZipDataset,
)
from monai.apps import CrossValidation
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.losses import DiceLoss, DiceCELoss
from monai.losses.ssim_loss import SSIMLoss
from monai.visualize import GradCAM
from monai.metrics import DiceMetric, SurfaceDiceMetric, HausdorffDistanceMetric, ROCAUCMetric, ConfusionMatrixMetric, MSEMetric
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    KeepLargestConnectedComponent,
    Compose,
    Resized,
    CropForegroundd,
    EnsureTyped,
    SqueezeDimd,
    FgBgToIndicesd,
    LoadImaged,
    MapLabelValued,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandFlipd,
    RandRotated,
    NormalizeIntensityd
)
from monai.utils import set_determinism
from utils import *
import wandb
from monai.networks.nets import BasicUNet
from einops import rearrange

class Trainer():
    def __init__(self,
                 method_name=None,
                 model=None,
                 train_val_path={'HCM': 'HCM/train_val_dataset.pkl', 'CAMUS': 'CAMUS/train_val_dataset.pkl',
                                'HCM-ext': 'HCM/train_val_dataset.pkl'},
                 test_path={'HCM': 'HCM/test_dataset.pkl', 'CAMUS': 'CAMUS/test_dataset.pkl', 'HCM-ext': 'HCM-ext/test_dataset.pkl'},
                 config=None,
                 debug=False,
            ):
        set_determinism(seed=0)
        set_track_meta(True)
        
        self.config = config
        self.method_name = method_name
        
        self.model = model
        self.init_model_state_dict = self.model.state_dict()
        
        self.logger = None
        self.debug = debug
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        self.train_val_path_dict = train_val_path
        self.test_path_dict = test_path


    def transformations(self, trans_type='train', device="cuda:0"):
        train_transforms = [
            LoadImaged(keys=["image", "label"], image_only=False),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=self.config.image_size, mode=("bilinear", "nearest")),
            # Spacingd(keys=["image", "label"], pixdim=(1.35, 1), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0).set_random_state(seed=0),
            # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
            EnsureTyped(keys=["image", "label", "view"], device=device, track_meta=True),
        ]

        val_or_test_transforms = [
            LoadImaged(keys=["image", "label"], image_only=False),
            # EnsureChannelFirstd(keys=["image", "label"]),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image", "label"], spatial_size=self.config.image_size, mode=("bilinear", "nearest")),
            # Spacingd(keys=["image", "label"], pixdim=(1.35, 1), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRanged(keys=["label"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
            # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
            EnsureTyped(keys=["image", "label", "view"], device=device, track_meta=True)
        ]

        plot_transforms = [
            LoadImaged(keys=["image", "label"], image_only=False),
            # EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"], spatial_size=self.config.image_size, mode=("bilinear", "nearest")),
        ]

        if trans_type == 'train':
            trans = Compose(train_transforms).set_random_state(seed=0)
        elif trans_type == 'val' or trans_type == 'test':
            trans = Compose(val_or_test_transforms).set_random_state(seed=0)
        elif trans_type == 'plot':
            trans = Compose(plot_transforms).set_random_state(seed=0)
        else:
            raise TypeError('Type can only be train, val or test')

        return trans

    def make_multi_views_train_val_dataset(self, train_val_path='HCM/train_val_dataset.pkl', fold=0):
        train_val_images = pickle.load(open(train_val_path, 'rb'))
        if 'CAMUS' in train_val_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_camus_info
        elif 'HCM' in train_val_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_hcm_info
        if self.config.frame_align == 'RA' or 'HMCQU' in train_val_path:
            frame_align_func = frame_align
        elif self.config.frame_align == 'FA' or 'CAMUS' in train_val_path:
            frame_align_func = frame_align_Full
        elif self.config.frame_align == 'ES':
            frame_align_func = frame_align_EDES
        if self.config.is_cv:
            def split_list_n_list(origin_list, n):
                if len(origin_list) % n == 0:
                    cnt = len(origin_list) // n
                else:
                    cnt = len(origin_list) // n + 1
            
                for i in range(0, n):
                    yield origin_list[i*cnt:(i+1)*cnt]
            
            folds = list(range(self.config.cross_validation_folds))
            partial = folds[0:fold] + folds[(fold + 1):]
            patients = list(train_val_images.keys())
            fold_patients = list(split_list_n_list(patients, self.config.cross_validation_folds))
            val_patients = fold_patients[fold]
            train_patients = []
            for i in partial:
                train_patients += fold_patients[i]
        else:
            patients = list(train_val_images.keys())
            train_patients = patients[:int(len(patients)*0.8)]
            val_patients = patients[int(len(patients)*0.8):]
        
        if self.config.is_video:
            train_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            val_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
                     
            for patient in train_patients:
                flag = True
                frames = {}
                for view in train_val_images[patient]:
                    if len(train_val_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = train_val_images[patient][view]
                if flag:
                    if self.config.mv:
                        if 'CAMUS' in train_val_path:
                            ed, es = read_info_func(info, patient.split('/')[-1])
                            frames = frame_align_func(frames, ed, None, es)
                        elif 'HCM' in train_val_path:
                            es = read_info_func(info, patient.split('/')[-1])
                            start = {0: 0, 1: 0, 2: 0}
                            ed = start
                            end = {0: len(frames[list(frames.keys())[0]]), 1: len(frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                            # frames = frame_align_func(frames, start, es, end)
                            #####
                            start = {k: 0 for k in frames.keys()}
                            es = {k: es[idx] for idx, k in enumerate(frames.keys())}
                            end = {k: len(frames[k]) for k in frames.keys()}
                            frames = preprocessing(frames, start, es, end)
                        else:
                            frames = frame_align_func(frames, None, None, None)
                    #### frames = stretch_videos(frames)
                    
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        train_image_temps[view].extend(frames[view])

            for patient in val_patients:
                flag = True
                frames = {}
                for view in train_val_images[patient]:
                    if len(train_val_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = train_val_images[patient][view]
                if flag:
                    if self.config.mv:
                        if 'CAMUS' in train_val_path:
                            ed, es = read_info_func(info, patient.split('/')[-1])
                            frames = frame_align_func(frames, ed, None, es)
                        elif 'HCM' in train_val_path:
                            es = read_info_func(info, patient.split('/')[-1])
                            start = {0: 0, 1: 0, 2: 0}
                            ed = start
                            end = {0: len(frames[list(frames.keys())[0]]), 1: len(frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                            # frames = frame_align_func(frames, start, es, end)
                            #####
                            start = {k: 0 for k in frames.keys()}
                            es = {k: es[idx] for idx, k in enumerate(frames.keys())}
                            end = {k: len(frames[k]) for k in frames.keys()}
                            frames = preprocessing(frames, start, es, end)
                        else:
                            frames = frame_align_func(frames, None, None, None)
                    ##### frames = stretch_videos(frames)
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        val_image_temps[view].extend(frames[view])
        else:
            train_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            val_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            for patient in train_patients:
                flag = True
                frames = {}
                for view in train_val_images[patient]:
                    if len(train_val_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = train_val_images[patient][view]
                if flag:
                    if 'CAMUS' in train_val_path:
                        ed, es = read_info_func(info, patient.split('/')[-1])
                        frames = frame_align_func(frames, ed, None, es)
                    elif 'HCM' in train_val_path:
                        es = read_info_func(info, patient.split('/')[-1])
                        start = {0: 0, 1: 0, 2: 0}
                        end = {0: len(frames[list(frames.keys())[0]]), 1: len(frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                        frames = frame_align_func(frames, start, es, end)
                    else:
                        frames = frame_align_func(frames, None, None, None)
                        
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        for image in frames[view]:
                            train_image_temps[view].append(image)
            for patient in val_patients:
                flag = True
                frames = {}
                for view in train_val_images[patient]:
                    if len(train_val_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = train_val_images[patient][view]
                if flag:
                    if 'CAMUS' in train_val_path:
                        ed, es = read_info_func(info, patient.split('/')[-1])
                        frames = frame_align_func(frames, ed, None, es)
                    elif 'HCM' in train_val_path:
                        es = read_info_func(info, patient.split('/')[-1])
                        start = {0: 0, 1: 0, 2: 0}
                        end = {0: len(frames[list(frames.keys())[0]]), 1: len(frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                        frames = frame_align_func(frames, start, es, end)
                    else:
                        frames = frame_align_func(frames, None, None, None)
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        for image in frames[view]:
                            val_image_temps[view].append(image)
        
        train_dss = []
        val_dss = []
        debug_dss = []
        # def flatten_dict(d):
        #     """
        #     将嵌套字典或列表拉平成一个列表。
        #     """
        #     result = []

        #     def _flatten(value):
        #         if isinstance(value, dict):
        #             for v in value.values():
        #                 _flatten(v)
        #         elif isinstance(value, list):
        #             for item in value:
        #                 _flatten(item)
        #         else:
        #             result.append(value)

        #     _flatten(d)
        #     return result
        # train_images_flatten = flatten_dict(train_image_temps)
        # np.save(f'train_images_flatten_fold{fold}.npy', train_images_flatten)
        # print(fold)
        # return None, None
        for view in train_image_temps.keys():
            train_images = train_image_temps[view]
            val_images = val_image_temps[view]
            if self.config.is_video:
                train_labels = [i[0].replace('image', 'label') for i in train_images]
                val_labels = [i[0].replace('image', 'label') for i in val_images]
            else:
                train_labels = [i.replace('image', 'label') for i in train_images]
                val_labels = [i.replace('image', 'label') for i in val_images]

            train_dicts = [{"image": image_name, 
                            "label": label_name, 
                            "view": [self.config.views_dict[self.config.dataset].index(view)]*len(image_name)
                            if self.config.is_video else self.config.views_dict[self.config.dataset].index(view)} for image_name, label_name in zip(train_images, train_labels)]

            
            val_dicts = [{"image": image_name, 
                         "label": label_name, 
                         "view": [self.config.views_dict[self.config.dataset].index(view)]*len(image_name) if self.config.is_video else self.config.views_dict[self.config.dataset].index(view), 
                          "name": label_name if 'HCM' in train_val_path else '/'.join(insert_and_return_value(label_name.split('/'), [1, 3], ['device', 'h']))} for image_name, label_name in zip(val_images, val_labels)]

            train_trans = self.transformations(trans_type='train', device=self.device)
            val_trans = self.transformations(trans_type='val', device=self.device)
            
            train_ds = CacheDataset(data=train_dicts, transform=train_trans, cache_rate=1.0, num_workers=8, copy_cache=False)
            val_ds = CacheDataset(data=val_dicts, transform=val_trans, cache_rate=1.0, num_workers=5, copy_cache=False)

            # TODO: Testing
            # train_ds = partition_dataset(train_ds, num_partitions=2, shuffle=True)[0]
            
            train_dss.append(train_ds)
            val_dss.append(val_ds)
            
            # debug_ds = CacheDataset(data=train_dicts, cache_rate=1.0, num_workers=8, copy_cache=False)
            # debug_dss.append(debug_ds)
        if self.config.mv:
            train_ds = ZipDataset(train_dss)
            val_ds = ZipDataset(val_dss)
        else:
            from torch.utils.data import ConcatDataset
            length = len(train_dss)
            train_ds = train_dss[0]
            val_ds = val_dss[0]
            for i in range(1, length):
                train_ds = ConcatDataset([train_ds, train_dss[i]])
                val_ds = ConcatDataset([val_ds, val_dss[i]])
            
        
        # debug_ds = ZipDataset(debug_dss)
        # debug_loader = ThreadDataLoader(debug_ds, num_workers=0, batch_size=4, shuffle=True)
        # debug_inter = iter(debug_loader)
        # for i in range(10):
        #     debug_batch = next(debug_inter)
        #     print(debug_batch)
        # exit()
            
        # disable multi-workers because `ThreadDataLoader` works with multi-threads
            
        return train_ds, val_ds

    def make_train_val_dataset(self, train_val_path='HCM/train_val_dataset.pkl', fold=0):
        train_val_images = pickle.load(open(train_val_path, 'rb'))

        if self.config.is_cv:
            def split_list_n_list(origin_list, n):
                if len(origin_list) % n == 0:
                    cnt = len(origin_list) // n
                else:
                    cnt = len(origin_list) // n + 1

                for i in range(0, n):
                    yield origin_list[i*cnt:(i+1)*cnt]

            folds = list(range(self.config.cross_validation_folds))
            partial = folds[0:fold] + folds[(fold + 1):]

            patients = list(train_val_images.keys())
            fold_patients = list(split_list_n_list(patients, self.config.cross_validation_folds))
            val_patients = fold_patients[fold]
            train_patients = []
            for i in partial:
                train_patients += fold_patients[i]
        else:
            patients = list(train_val_images.keys())
            train_patients = patients[:int(len(patients)*0.8)]
            val_patients = patients[int(len(patients)*0.8):]

        if self.config.is_video:
            train_temps = []
            val_temps = []
            train_views = []
            val_views = []
            for patient in train_patients:
                for view in train_val_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    if len(train_val_images[patient][view]) == 0:
                        continue
                    slide_windows = slide_window(len(train_val_images[patient][view]), window_size=self.config.n_segment, overlap=self.config.n_overlap)
                    for window in slide_windows:
                        p_temps = []
                        view_temps = []
                        for j in range(window[0], window[1]):
                            p_temps.append(train_val_images[patient][view][j])
                            view_temps.append(self.config.views_dict[self.config.dataset].index(view))
                        train_temps.append(p_temps)
                        train_views.append(view_temps)

            for patient in val_patients:
                for view in train_val_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    if len(train_val_images[patient][view]) == 0:
                        continue
                    slide_windows = slide_window(len(train_val_images[patient][view]), window_size=self.config.n_segment, overlap=self.config.n_overlap)
                    for window in slide_windows:
                        p_temps = []
                        view_temps = []
                        for j in range(window[0], window[1]):
                            p_temps.append(train_val_images[patient][view][j])
                            view_temps.append(self.config.views_dict[self.config.dataset].index(view))
                        val_temps.append(p_temps)
                        val_views.append(view_temps)

            train_images = train_temps
            train_labels = [[i.replace('image', 'label') for i in j] for j in train_images]

            val_images = val_temps
            val_labels = [[i.replace('image', 'label') for i in j] for j in val_images]
        else:
            train_temps = []
            val_temps = []
            train_views = []
            val_views = []
            # for classifcation
            for patient in train_patients:
                for view in train_val_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    for image in train_val_images[patient][view]:
                        train_views.append(self.config.views_dict[self.config.dataset].index(view))
                        train_temps.append(image)

            for patient in val_patients:
                for view in train_val_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    for image in train_val_images[patient][view]:
                        val_views.append(self.config.views_dict[self.config.dataset].index(view))
                        val_temps.append(image)

            train_images = train_temps
            # train_images = np.load(f'train_images_flatten_fold{fold}.npy', allow_pickle=True)
            # print(len(train_images))
            train_labels = [i.replace('image', 'label') for i in train_images]
            # print(len(train_labels))

            val_images = val_temps
            val_labels = [i.replace('image', 'label') for i in val_images]

            # for patient in train_patients:
            #     for view in train_val_images[patient]:
            #         for image in train_val_images[patient][view]:
            #             train_temps.append(image)

            # for patient in val_patients:
            #     for view in train_val_images[patient]:
            #         for image in train_val_images[patient][view]:
            #             val_temps.append(image)

            # train_images = train_temps
            # train_labels = [i.replace('image', 'label') for i in train_images]

            # val_images = val_temps
            # val_labels = [i.replace('image', 'label') for i in val_images]
        # train_dicts = [{"image": image_name, "label": label_name, "view": 0}
        # for image_name, label_name in zip(train_images, train_labels)]
        train_dicts = [{"image": image_name, "label": label_name, "view": view} for image_name, label_name, view in zip(train_images, train_labels, train_views)]
        val_dicts = [{"image": image_name, "label": label_name, "view": view} for image_name, label_name, view in zip(val_images, val_labels, val_views)]
        print(len(train_dicts))
        train_trans = self.transformations(trans_type='train', device=self.device)
        val_trans = self.transformations(trans_type='val', device=self.device)

        train_ds = CacheDataset(data=train_dicts, transform=train_trans, cache_rate=1.0, num_workers=8, copy_cache=False)
        val_ds = CacheDataset(data=val_dicts, transform=val_trans, cache_rate=1.0, num_workers=5, copy_cache=False)

        return train_ds, val_ds

    def make_test_dataset(self, test_path='HCM/test_dataset.pkl'):
        test_images = pickle.load(open(test_path, 'rb'))
        patients = list(test_images.keys())
        if self.config.is_video:
            test_temps = []
            test_views = []
            for patient in patients:
                for view in test_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    if len(test_images[patient][view]) == 0:
                        continue
                    slide_windows = slide_window(len(test_images[patient][view]), window_size=self.config.n_segment, overlap=self.config.n_overlap)
                    for window in slide_windows:
                        p_temps = []
                        view_temps = []
                        for j in range(window[0], window[1]):
                            p_temps.append(test_images[patient][view][j])
                            view_temps.append(self.config.views_dict[self.config.dataset].index(view))
                        test_temps.append(p_temps)
                        test_views.append(view_temps)

            test_images = test_temps
            test_labels = [[i.replace('image', 'label') for i in j] for j in test_images]

        else:
            test_temps = []
            test_views = []
            for patient in patients:
                for view in test_images[patient]:
                    if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                        continue
                    for image in test_images[patient][view]:
                        test_temps.append(image)
                        test_views.append(self.config.views_dict[self.config.dataset].index(view))

            test_images = test_temps
            test_labels = [i.replace('image', 'label') for i in test_images]

        test_dicts = [{"image": image_name, "label": label_name, "view": view, "name": label_name if 'HCM' in test_path else '/'.join(insert_and_return_value(label_name.split('/'), [1, 3], ['device', 'h']))} for image_name, label_name, view in zip(test_images, test_labels, test_views)]

        test_trans = self.transformations(trans_type='test', device=self.device)

        test_ds = CacheDataset(data= test_dicts, transform=test_trans, cache_rate=1.0, num_workers=5, copy_cache=False)

        return test_ds

    def make_multi_views_test_dataset(self, test_path='HCM/test_dataset.pkl'):
        test_images = pickle.load(open(test_path, 'rb'))
        patients = list(test_images.keys())
        if 'CAMUS' in test_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_camus_info
        elif 'HCM' in test_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_hcm_info
        if self.config.frame_align == 'RA' or 'HMCQU' in test_path:
            frame_align_func = frame_align
        elif self.config.frame_align == 'FA' or 'CAMUS' in test_path:
            frame_align_func = frame_align_Full
        elif self.config.frame_align == 'ES':
            frame_align_func = frame_align_EDES
        
        
        if self.config.is_video:
            test_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            for patient in patients:
                flag = True
                frames = {}
                for view in test_images[patient]:
                    if len(test_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = test_images[patient][view]
                if flag:
                    if self.config.mv:
                        if 'CAMUS' in test_path:
                            ed, es = read_info_func(info, patient.split('/')[-1])
                            frames = frame_align_Full(frames, ed, None, es)
                        elif 'HCM' in test_path:
                            es = read_info_func(info, patient.split('/')[-1])
                            start = {0: 0, 1: 0, 2: 0}
                            end = {0: len(frames[list(frames.keys())[0]]), 1: len(
                                frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                            # frames = frame_align_func(frames, start, es, end)
                            #####
                            start = {k: 0 for k in frames.keys()}
                            es = {k: es[idx] for idx, k in enumerate(frames.keys())}
                            end = {k: len(frames[k]) for k in frames.keys()}
                            frames = preprocessing(frames, start, es, end)
                        else:
                            frames = frame_align_func(frames, None, None, None)
                    
                    #### frames = stretch_videos(frames)
                    
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        test_image_temps[view].extend(frames[view])
        else:
            test_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            for patient in patients:
                flag = True
                frames = {}
                for view in test_images[patient]:
                    if len(test_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = test_images[patient][view]
                if flag:
                    if 'CAMUS' in test_path:
                        ed, es = read_info_func(info, patient.split('/')[-1])
                        frames = frame_align_Full(frames, ed, None, es)
                    elif 'HCM' in test_path:
                        es = read_info_func(info, patient.split('/')[-1])
                        start = {0: 0, 1: 0, 2: 0}
                        end = {0: len(frames[list(frames.keys())[0]]), 1: len(
                            frames[list(frames.keys())[1]]), 2: len(frames[list(frames.keys())[2]])}
                        frames = frame_align_func(frames, start, es, end)
                    else:
                        frames = frame_align_func(frames)
                        
                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        for image in frames[view]:
                            test_image_temps[view].append(image)

        test_dss = []
        for view in test_image_temps.keys():
            test_images = test_image_temps[view]
            if self.config.is_video:
                test_labels = [i[0].replace('image', 'label') for i in test_images]
            else:
                test_labels = [i.replace('image', 'label') for i in test_images]
            
            test_dicts = [{"image": image_name, 
                        "label": label_name, 
                        "view": [self.config.views_dict[self.config.dataset].index(view)]*len(image_name) 
                        if self.config.is_video else self.config.views_dict[self.config.dataset].index(view), 
                        "name": label_name if 'HCM' in test_path else '/'.join(insert_and_return_value(label_name.split('/'), [1, 3], ['device', 'h']))} for image_name, label_name in zip(test_images, test_labels)]

            test_trans = self.transformations(trans_type='test', device=self.device)

            test_ds = CacheDataset(data=test_dicts, transform=test_trans, cache_rate=1.0, num_workers=5, copy_cache=False)
            test_dss.append(test_ds)
        
        if self.config.mv:
            test_ds = ZipDataset(test_dss)
        else:
            from torch.utils.data import ConcatDataset
            length = len(test_dss)
            test_ds = test_dss[0]
            for i in range(1, length):
                test_ds = ConcatDataset([test_ds, test_dss[i]])
        
        return test_ds

    def calculate_loss(self, outputs, labels, pred_views=None, views=None):
        if self.config.multi_output:
            dice_loss = torch.tensor(0.0).to(self.device)
                    
            if self.config.ssim_loss:
                ssim_loss = torch.tensor(0.0).to(self.device)
                        
            for out in outputs:
                dice_loss += self.dice_loss_function(out, labels)
                if self.config.ssim_loss:
                    ssim_loss += self.ssim_loss_function(torch.argmax(out, dim=1, keepdim=True), labels)

            dice_loss /= len(outputs)
            if self.config.ssim_loss:
                ssim_loss /= len(outputs)
                loss = dice_loss + ssim_loss
            else:
                loss = dice_loss
            
        else:
            dice_loss = self.dice_loss_function(outputs, labels)
            loss = dice_loss

            if self.config.co_classify:
                ce_loss = self.co_classify_loss_function(pred_views.squeeze(-1).squeeze(-1), views.squeeze())
                loss += ce_loss
                            
            if self.config.ssim_loss:
                ssim_loss = self.ssim_loss_function(torch.argmax(outputs, dim=1, keepdim=True), labels)
                loss += ssim_loss
                
        return loss

    def train(self):
        set_track_meta(False)

        mean_dices = []
        mean_nsds = []
        mean_hd95s = []
        std_hd95s = []
        std_dices = []
        std_nsds = []
        
        begin_fold = 0
        # end_fold = 2
        end_fold = self.config.cross_validation_folds
        
        for fold in range(begin_fold, end_fold):
            if not self.debug:
                wandb.init(project=self.method_name, config=self.config)
            if self.config.ext_test:
                self.save_path = os.path.join(self.config.save_path, 'HCM', self.method_name)
            else:
                self.save_path = os.path.join(self.config.save_path, self.config.dataset, self.method_name)
            self.train_val_path = self.train_val_path_dict[self.config.dataset]
            self.test_path = self.test_path_dict[self.config.dataset]
            fold_save_path = os.path.join(self.save_path, f'fold{fold}')
            monai_start = time.time()
            '''
                -------------train--------------
            '''
            if self.config.multi_view:
                train_ds, val_ds = self.make_multi_views_train_val_dataset(self.train_val_path, fold)
            else:
                train_ds, val_ds = self.make_train_val_dataset(self.train_val_path, fold)
            self.train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=self.config.batch_size, shuffle=True)
            self.val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
        
            self.len_train_ds = len(train_ds)
            self.len_val_ds = len(val_ds)
            
            if not os.path.exists(fold_save_path):
                os.makedirs(fold_save_path)
            if self.logger is not None:
                self.logger.delete()
            self.logger = Logger(fold_save_path, self.method_name)
            self.logger.log(
                "----------------------------------"
                f"{fold + 1}th fold training begin"
                "----------------------------------"
            )
            self.model.load_state_dict(self.init_model_state_dict)
            if not self.debug:
                wandb.watch(self.model, log="all", log_freq=50)
            (   
                epoch_num,
                m_epoch_loss_values,
                m_metric_values,
                m_epoch_times,
                m_train_time,
            ) = self.train_process_mv(fold_index=fold) if self.config.multi_view else self.train_process(fold_index=fold)
            m_total_time = time.time() - monai_start
            self.logger.log(
                f"{fold + 1}th fold total time of {epoch_num} epochs with MONAI fast training: {m_train_time:.4f},"
                f" time of preparing cache: {(m_total_time - m_train_time):.4f}"
            )
            np.save(os.path.join(self.save_path, f'fold{fold}', f"{self.method_name}_epoch_loss_values.npy"), np.array(m_epoch_loss_values))
            np.save(os.path.join(self.save_path, f'fold{fold}', f"{self.method_name}_metric_values.npy"), np.array(m_metric_values))
            np.save(os.path.join(self.save_path, f'fold{fold}', f"{self.method_name}_epoch_times.npy"), np.array(m_epoch_times))
            np.save(os.path.join(self.save_path, f'fold{fold}', f"{self.method_name}_total_time.npy"), np.array(m_total_time))
            
            '''
                -------------test--------------
            '''
            if self.config.ed_test:
                test_ds = self.make_multi_views_edes_test_dataset(self.test_path, edores='ed')
            elif self.config.es_test:
                test_ds = self.make_multi_views_edes_test_dataset(self.test_path, edores='es')
            elif self.config.multi_view:
                test_ds = self.make_multi_views_test_dataset(self.test_path)
            else:
                test_ds = self.make_test_dataset(self.test_path)
            
            self.test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)
    
            self.len_test_ds = len(test_ds)
            
            if self.logger is not None:
                self.logger.delete()
            if self.config.ed_test:
                self.logger = Logger(fold_save_path, self.method_name + '_edtest')
            elif self.config.es_test:
                self.logger = Logger(fold_save_path, self.method_name + '_estest')
            elif self.config.ext_test:
                self.logger = Logger(fold_save_path, self.method_name + '_exttest')
            else:
                self.logger = Logger(fold_save_path, self.method_name + '_test')
            self.logger.log(
                "----------------------------------"
                f"{fold + 1}th fold testing begin"
                "----------------------------------"
            )
            self.model.load_state_dict(self.init_model_state_dict)
            self.model.load_state_dict(torch.load(os.path.join(fold_save_path, f"{self.method_name}_best.pt")))
            begin_time = time.time()
            mean_dice, mean_nsd, mean_hd95, std_dice, std_nsd, std_hd95 = self.test_process_mv(fold) if self.config.multi_view else self.test_process(fold)
            end_time = time.time()
            print(f"fold {fold + 1} test time: {end_time - begin_time}")
            # dice, nsd, hd95 = self.test_process_plot()
            mean_dices.append(mean_dice)
            mean_nsds.append(mean_nsd)
            mean_hd95s.append(mean_hd95)
            std_dices.append(std_dice)
            std_nsds.append(std_nsd)
            std_hd95s.append(std_hd95)
            wandb.finish()
            # exit()
            
        self.logger.log(f"{self.config.cross_validation_folds} folds CV finish! {self.config.cross_validation_folds} models mean dice: {np.mean(mean_dices):.4f}±{np.mean(std_dices):.4f} mean nsd: {np.mean(mean_nsds):.4f}±{np.mean(std_nsds):.4f} mean hd95: {np.mean(mean_hd95s):.4f}±{np.mean(std_hd95s):.4f}")
        
    def train_process(self, fold_index):
        self.dice_loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            batch=False,
            smooth_nr=0.00001,
            smooth_dr=0.00001,
            lambda_dice=0.5,
            lambda_ce=0.5,
        )
        
        if self.config.ssim_loss:
            self.ssim_loss_function = SSIMLoss(spatial_dims=2)
        
        if self.config.co_classify:
            self.co_classify_loss_function = torch.nn.CrossEntropyLoss()
        
        self.model.to(self.device)
            
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        post_pred_view = Compose([AsDiscrete(argmax=True, to_onehot=len(self.config.views_dict[self.config.dataset]))])
        post_view = Compose([AsDiscrete(to_onehot=len(self.config.views_dict[self.config.dataset]))])

        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        nsd_metric = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, class_thresholds=self.config.class_thresholds)
        hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False, percentile=95)
        if self.config.co_classify:
            acc_metric = ConfusionMatrixMetric(include_background=True, metric_name=['accuracy', 'f1 score'], reduction="mean", get_not_nans=False)
        
        if self.config.optimizer == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        
        if self.config.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config.T0, T_mult=self.config.Tmult, eta_min=1e-5, last_epoch=-1)
        elif self.config.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
        else:
            scheduler = None
        
        if self.config.is_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        if self.config.co_classify:
            best_metric = (-1, -1, 999, -1, -1)
        else:
            best_metric = (-1, -1, 999)
            
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        epoch_times = []
        total_start = time.time()
        patience = 0
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            self.logger.log("-" * 10)
            self.logger.log(f"epoch {epoch + 1}/{self.config.max_epochs}")

            self.model.train()
            epoch_loss = 0
            train_loader_iterator = iter(self.train_loader)

            # using step instead of iterate through train_loader directly to track data loading time
            # steps are 1-indexed for printing and calculation purposes
            for step in range(1, len(self.train_loader) + 1):
                step_start = time.time()
                batch_data = next(train_loader_iterator)

                if self.config.multi_view:
                    input = []
                    label = []
                    view = []
                    for i_view in range(len(self.config.views)):
                        input.append(batch_data[i_view]["image"].to(self.device).unsqueeze(1))
                        label.append(batch_data[i_view]["label"].to(self.device).unsqueeze(1))
                        view.append(batch_data[i_view]["view"].to(self.device).unsqueeze(1))
  
                    inputs = torch.cat(input, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    labels = torch.cat(label, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    views = torch.cat(view, dim=1).view(-1, 1)
                else:
                    inputs, labels, views = (
                            batch_data["image"].to(self.device),
                            batch_data["label"].to(self.device),
                            batch_data["view"].to(self.device),
                        )
                    if self.config.is_video:
                        inputs = inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        labels = labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        views = views.view(-1, 1)
                    # print(inputs.shape, labels.shape, views.shape)
                # plot_3d_mat(inputs.to(dtype=torch.float32), epoch, self.save_path, name='image')
                # plot_3d_mat(labels.to(dtype=torch.float32), step, self.save_path, name='label')
                optimizer.zero_grad()

                if self.config.is_amp:
                    with torch.cuda.amp.autocast():
                        if self.config.co_classify:
                            outputs, pred_views = self.model(inputs)
                            loss = self.calculate_loss(outputs, labels, pred_views, views)
                        else:
                            outputs = self.model(inputs)
                            loss = self.calculate_loss(outputs, labels)
                                
                else:
                    if self.config.co_classify:
                        outputs, pred_views = self.model(inputs)
                        loss = self.calculate_loss(outputs, labels, pred_views, views)
                    else:
                        outputs = self.model(inputs)
                        loss = self.calculate_loss(outputs, labels)
                        
                # plot_3d_mat(torch.argmax(outputs, dim=1, keepdim=True).to(dtype=torch.float32), epoch, self.save_path, name='pred')
                
                if self.config.is_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_len = math.ceil(self.len_train_ds / self.train_loader.batch_size)
                if step % 50 == 1:
                    self.logger.log(
                        f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" f" step time: {(time.time() - step_start):.4f}"
                    )
            if scheduler is not None:
                scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            if not self.debug:
                wandb.log({"train_loss": epoch_loss})
            self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.config.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loader_iterator = iter(self.val_loader)

                    for _ in range(len(self.val_loader)):
                        val_data = next(val_loader_iterator)
                        if self.config.multi_view:
                            input = []
                            label = []
                            view = []
                            for i_view in range(len(self.config.views)):
                                input.append(val_data[i_view]["image"].to(self.device).unsqueeze(1))
                                label.append(val_data[i_view]["label"].to(self.device).unsqueeze(1))
                                view.append(val_data[i_view]["view"].to(self.device).unsqueeze(1))
                    
                            val_inputs = torch.cat(input, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                            val_labels = torch.cat(label, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                            val_views = torch.cat(view, dim=1).view(-1, 1)

                        else:
                            val_inputs, val_labels, val_views = (
                                val_data["image"].to(self.device),
                                val_data["label"].to(self.device),
                                val_data["view"].to(self.device),
                            )
                        
                            if self.config.is_video:
                                val_inputs = val_inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                                val_labels = val_labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                                val_views = val_views.view(-1, 1)
                        
                        if self.config.is_amp:
                            with torch.cuda.amp.autocast():
                                if self.config.co_classify:
                                    val_outputs, val_pred_views = self.model(val_inputs)
                                else:
                                    val_outputs = self.model(val_inputs)
                            
                                if self.config.multi_output:
                                    val_outputs = val_outputs[-1]

                                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                                if self.config.co_classify:
                                    val_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(val_pred_views.squeeze(-1).squeeze(-1))]
                                    val_views = [post_view(i).to(self.device) for i in decollate_batch(val_views)]

                                dice_metric(y_pred=val_outputs, y=val_labels)
                                nsd_metric(y_pred=val_outputs, y=val_labels)
                                hd95_metric(y_pred=val_outputs, y=val_labels)
                                if self.config.co_classify:
                                    acc_metric(y_pred=val_pred_views, y=val_views)
                            
                        else:
                            if self.config.co_classify:
                                val_outputs, val_pred_views = self.model(val_inputs)
                            else:
                                val_outputs = self.model(val_inputs)
                                
                            if self.config.multi_output:
                                val_outputs = val_outputs[-1]

                            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                            if self.config.co_classify:
                                val_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(val_pred_views.squeeze(-1).squeeze(-1))]
                                val_views = [post_view(i).to(self.device) for i in decollate_batch(val_views)]

                            dice_metric(y_pred=val_outputs, y=val_labels)
                            nsd_metric(y_pred=val_outputs, y=val_labels)
                            hd95_metric(y_pred=val_outputs, y=val_labels)
                            if self.config.co_classify:
                                acc_metric(y_pred=val_pred_views, y=val_views)
                        
                    dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                    
                    nsd = nsd_metric.aggregate().item()
                    nsd_metric.reset()
                    
                    hd95 = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                    
                    if self.config.co_classify:
                        confusion = acc_metric.aggregate()
                        acc_metric.reset()
                        acc, f1 = confusion[0].item(), confusion[1].item()
                        metric_values.append((dice, nsd, hd95, acc, f1))
                    else:
                        metric_values.append((dice, nsd, hd95))
                    
                    if hd95 == 0:
                        hd95 = 999
                    if not self.debug:
                        wandb.log({"val_dice": dice, "val_nsd": nsd, "val_hd95": hd95})
                    
                    if (dice + nsd - hd95 * 0.04) > (best_metric[0] + best_metric[1] - best_metric[2] * 0.04) and patience < self.config.early_stopping_patience:
                        if self.config.co_classify:
                            best_metric = (dice, nsd, hd95, acc, f1)
                        else:
                            best_metric = (dice, nsd, hd95)
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'fold{fold_index}', f"{self.method_name}_best.pt"))
                        self.logger.log("saved new best metric model")
                        patience = 0
                    elif patience >= self.config.early_stopping_patience:
                        self.logger.log("early stopping patience reached")
                        break
                    else:
                        self.logger.log("current epoch patience is {}".format(patience))
                        patience += 1

                    self.logger.log(
                        f"current epoch: {epoch + 1} current"
                        f" mean dice: {dice:.4f}"
                        f" mean nsd: {nsd:.4f}"
                        f" mean hd95: {hd95:.4f}"
                    )
                    if self.config.co_classify:
                        self.logger.log(
                            f" mean acc: {acc:.4f}"
                            f" mean f1: {f1:.4f}"
                        )
                    self.logger.log(
                        f" best mean dice: {best_metric[0]:.4f}"
                        f" best mean nsd: {best_metric[1]:.4f}"
                        f" best mean hd95: {best_metric[2]:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
                    if self.config.co_classify:
                        self.logger.log(
                            f" mean acc: {acc:.4f}"
                            f" mean f1: {f1:.4f}"
                            f" best mean acc: {best_metric[3]:.4f}"
                            f" best mean f1: {best_metric[4]:.4f}"
                        )
                
            self.logger.log(f"time consuming of epoch {epoch + 1} is:" f" {(time.time() - epoch_start):.4f}")
            epoch_times.append(time.time() - epoch_start)

        total_time = time.time() - total_start
        self.logger.log(
            f"train completed, best_metric: {best_metric[0]:.4f} {best_metric[1]:.4f} {best_metric[2]:.4f}"
            f" at epoch: {best_metric_epoch}"
            f" total time: {total_time:.4f}"
        )
        return (
            epoch + 1,
            epoch_loss_values,
            metric_values,
            epoch_times,
            total_time,
        )
        
    def test(self):
        set_track_meta(False)
        dices = []
        nsds = []
        hd95s = []
        if self.config.multi_view:
            test_ds = self.make_multi_views_test_dataset(self.test_path)
        else:
            test_ds = self.make_test_dataset(self.test_path)

        self.test_loader = ThreadDataLoader(test_ds, num_workers=0, batch_size=1)
        self.len_test_ds = len(test_ds)
        
        for fold in range(self.config.cross_validation_folds):
            fold_save_path = os.path.join(self.save_path, f'fold{fold}')
            if self.logger is not None:
                self.logger.delete()
            self.logger = Logger(fold_save_path, self.method_name + '_test')
            self.logger.log(
                "----------------------------------"
                f"{fold + 1}th fold testing begin"
                "----------------------------------"
            )
            self.model.load_state_dict(torch.load(os.path.join(fold_save_path, f"{self.method_name}_best.pt")))
            dice, nsd, hd95 = self.test_process()
            dices.append(dice)
            nsds.append(nsd)
            hd95s.append(hd95)
            
        self.logger.log(f"test finish! {self.config.cross_validation_folds} models mean dice: {np.mean(dices):.4f} mean nsd: {np.mean(nsds):.4f} mean hd95: {np.mean(hd95s):.4f}")
    
    def log_test_result(self, images, preds, labels, dice, nsd, hd95):
        table = wandb.Table(columns=["image", "pred", "label", "dice", "nsd", "hd95"])
        for img, pred, label, d, n, h in zip(images, preds, labels, dice, nsd, hd95):
            img = img[0].cpu().numpy()*255
            pred = pred.argmax(dim=0).cpu().numpy()*255
            label = label.argmax(dim=0).cpu().numpy()*255
            d = d.cpu().numpy()
            n = n.cpu().numpy()
            h = h.cpu().numpy()
            
            table.add_data(wandb.Image(img), wandb.Image(pred), wandb.Image(label), d, n, h)
        wandb.log({"test_table":table})
        
    def test_process(self, fold_index):
        self.model.to(self.device)
        
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        post_pred_view = Compose([AsDiscrete(argmax=True, to_onehot=len(self.config.views_dict[self.config.dataset]))])
        post_view = Compose([AsDiscrete(to_onehot=len(self.config.views_dict[self.config.dataset]))])

        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        nsd_metric = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, class_thresholds=self.config.class_thresholds)
        hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False, percentile=95)
        if self.config.co_classify:
            acc_metric = ConfusionMatrixMetric(include_background=True, metric_name=['accuracy', 'f1 score'], reduction="mean", get_not_nans=False)

        min_dice = (2, -1)
        min_nsd = (2, -1)
        max_hd95 = (0, -1)
        plot_data = [None, None, None]
        total_start = time.time()
        self.model.eval()
        with torch.no_grad():
            test_loader_iterator = iter(self.test_loader)
            patient_metrics = {}
            for idx in range(len(self.test_loader)):
                test_data = next(test_loader_iterator)
                if self.config.multi_view:
                    input = []
                    label = []
                    view = []
                    for i_view in range(len(self.config.views)):
                        input.append(test_data[i_view]["image"].to(self.device).unsqueeze(1))
                        label.append(test_data[i_view]["label"].to(self.device).unsqueeze(1))
                        view.append(test_data[i_view]["view"].to(self.device).unsqueeze(1))
                    test_inputs = torch.cat(input, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    test_labels = torch.cat(label, dim=1).view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    test_views = torch.cat(view, dim=1).view(-1, 1)
                else:
                    test_inputs, test_labels, test_views, name = (
                        test_data["image"].to(self.device),
                        test_data["label"].to(self.device),
                        test_data["view"].to(self.device),
                        test_data["name"],
                    )
                    
                    if self.config.is_video:
                        test_inputs = test_inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        test_labels = test_labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        test_views = test_views.view(-1, 1)
                    
                # plot_3d_mat(test_inputs.to(dtype=torch.float32), 0, fold_save_path, name='image')
                # plot_3d_mat(test_labels.to(dtype=torch.float32), 0, fold_save_path, name='label')
                if self.config.is_amp:
                    with torch.cuda.amp.autocast():
                        if self.config.co_classify:
                            test_outputs, test_pred_views = self.model(test_inputs)
                        else:
                            test_outputs = self.model(test_inputs)
                else:
                    if self.config.co_classify:
                        test_outputs, test_pred_views = self.model(test_inputs)
                    else:
                        test_outputs = self.model(test_inputs)
                        
                if self.config.multi_output:
                    test_outputs = test_outputs[-1]
                
                # plot_3d_mat(torch.argmax(test_outputs, dim=1, keepdim=True).to(dtype=torch.float32), 0, fold_save_path, name='pred')
                test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
                test_labels = [post_label(i) for i in decollate_batch(test_labels)]
                if self.config.co_classify:
                    test_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(test_pred_views.squeeze(-1).squeeze(-1))]
                    test_views = [post_view(i).to(self.device) for i in decollate_batch(test_views)]
                
                
                # plot_test_result(test_outputs[0].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join(f'methods_pred/{self.config.dataset}', f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[0].split("/")[2]}', f'{name[0].split("/")[4]}'), name=f'pred{name[0].split("label")[-1].split(".png")[0]}', cmap='gray')
                
                dice_ = dice_metric(y_pred=test_outputs, y=test_labels)
                nsd_ = nsd_metric(y_pred=test_outputs, y=test_labels)
                hd95_ = hd95_metric(y_pred=test_outputs, y=test_labels)
                if self.config.co_classify:
                    confusion_ = acc_metric(y_pred=test_pred_views, y=test_views)

                if name[0].split("/")[2] not in patient_metrics:
                    patient_metrics[name[0].split("/")[2]] = {}
                if name[0].split("/")[4] not in patient_metrics[name[0].split("/")[2]]:
                    patient_metrics[name[0].split("/")[2]][name[0].split("/")[4]] = {}
                if name[0].split("/")[-1] in patient_metrics[name[0].split("/")[2]][name[0].split("/")[4]].keys():
                    old_dice, old_nsd, old_hd95 = patient_metrics[name[0].split("/")[2]][name[0].split("/")[4]][name[0].split("/")[-1]]
                    if old_dice < dice_.item():
                        patient_metrics[name[0].split("/")[2]][name[0].split("/")[4]][name[0].split(
                            "/")[-1]] = (dice_.item(), nsd_.item(), hd95_.item())
                        # plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join('test_pred_result', f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[i][0].split("/")[2]}', f'{name[i][0].split("/")[4]}'), name=f'pred{name[i][0].split("label")[-1].split(".png")[0]}', cmap='gray')
                else:
                    patient_metrics[name[0].split("/")[2]][name[0].split("/")[4]][name[0].split(
                        "/")[-1]] = (dice_.item(), nsd_.item(), hd95_.item())
                    # plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join(
                    #     'test_pred_result', f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[i][0].split("/")[2]}', f'{name[i][0].split("/")[4]}'), name=f'pred{name[i][0].split("label")[-1].split(".png")[0]}', cmap='gray')
         
                # if idx < self.config.NUM_BATCHES_TO_LOG:
                #     self.log_test_result(test_inputs, test_outputs, test_labels, dice_, nsd_, hd95_)
                    
                if not self.config.is_video:
                    if dice_.min() < min_dice[0]:
                        min_dice = (dice_.min(), idx)
                        plot_data[0] = (test_outputs[0], test_labels[0], test_inputs)
                    if nsd_.min() < min_nsd[0]:
                        min_nsd = (nsd_.min(), idx)
                        plot_data[1] = (test_outputs[0], test_labels[0], test_inputs)
                    if hd95_.max() > max_hd95[0]:
                        max_hd95 = (hd95_.max(), idx)
                        plot_data[2] = (test_outputs[0], test_labels[0], test_inputs)
            
            # # 创建一个空的列表来存储行数据
            rows = []

            # 将数据字典转换为列表
            for patient_id, views in patient_metrics.items():
                for view, frames in views.items():
                    for frame, metrics in frames.items():
                        dice, nsd, hd95 = metrics
                        rows.append({
                            'Patient ID': patient_id,
                            'View': view,
                            'Frame': frame,
                            'Dice': dice,
                            'NSD': nsd,
                            'HD95': hd95
                        })

            # 将列表转换为DataFrame
            df = pd.DataFrame(rows, columns=['Patient ID',
                            'View', 'Frame', 'Dice', 'NSD', 'HD95'])

            # 保存DataFrame到Excel文件
            df.to_excel(f'patient_data_{self.config.project_name}.xlsx', index=False)
            # exit()
            metrics_dict = get_mean_and_std_metrics(patient_metrics)
            new_dice = []
            new_nsd = []
            new_hd95 = []
            for k, v in metrics_dict.items():
                new_dice.append(v[0])
                new_nsd.append(v[1])
                new_hd95.append(v[2])
            # dice = dice_metric.aggregate().item()
            dice_metric.reset()
            
            # nsd = nsd_metric.aggregate().item()
            nsd_metric.reset()
            
            # hd95 = hd95_metric.aggregate().item()
            hd95_metric.reset()
            
            if self.config.co_classify:
                confusion = acc_metric.aggregate()
                acc_metric.reset()
                acc, f1 = confusion[0].item(), confusion[1].item()
            
            total_time = time.time() - total_start
            
            if self.config.co_classify:
                self.logger.log(
                    f" mean dice: {np.mean(new_dice):.4f}"
                    f" mean nsd: {np.mean(new_nsd):.4f}"
                    f" mean hd95: {np.mean(new_hd95):.4f}"
                    f" mean acc: {acc:.4f}"
                    f" mean f1: {f1:.4f}"
                    f" total time: {total_time:.4f}"
                )
            else:
                self.logger.log(
                    f" mean dice: {np.mean(new_dice):.4f}"
                    f" mean nsd: {np.mean(new_nsd):.4f}"
                    f" mean hd95: {np.mean(new_hd95):.4f}"
                    f" total time: {total_time:.4f}"
                )
            if not self.config.is_video:
                self.logger.log(
                    f" min dice: {min_dice[0]:.4f} in {min_dice[1]}th batch |"
                    f" min nsd: {min_nsd[0]:.4f} in {min_nsd[1]}th batch |"
                    f" max hd95: {max_hd95[0]:.4f} in {max_hd95[1]}th batch"
                )
            
            test_metric = (np.mean(new_dice), np.mean(new_nsd), np.mean(
                new_hd95), np.std(new_dice), np.std(new_nsd), np.std(new_hd95))
            
            
        # if not self.config.is_video:
        #     np_plot_data = []
        #     for plot in plot_data:
        #         a = plot[0].argmax(dim=0).squeeze().transpose(0, 1).cpu().numpy()
        #         b = plot[1].argmax(dim=0).squeeze().transpose(0, 1).cpu().numpy()
        #         c = plot[2].squeeze().transpose(0, 1).cpu().numpy()
        #         a = resample(a, mode='nearest')
        #         b = resample(b, mode='nearest')
        #         c = resample(c, mode='linear')
        #         np_plot_data.append((a, b, c))
            
        #     plot_example(fold_save_path, np_plot_data, min_dice, min_nsd, max_hd95)
        
        return test_metric
    
    def train_process_mv(self, fold_index):
        self.dice_loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            batch=False,
            smooth_nr=0.00001,
            smooth_dr=0.00001,
            lambda_dice=0.5,
            lambda_ce=0.5,
        )
        
        if self.config.ssim_loss:
            self.ssim_loss_function = SSIMLoss(spatial_dims=2)
        
        if self.config.co_classify:
            self.co_classify_loss_function = torch.nn.CrossEntropyLoss()
        
        self.model.to(self.device)
        
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        post_pred_view = Compose([AsDiscrete(argmax=True, to_onehot=len(self.config.views_dict[self.config.dataset]))])
        post_view = Compose([AsDiscrete(to_onehot=len(self.config.views_dict[self.config.dataset]))])

        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        nsd_metric = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, class_thresholds=self.config.class_thresholds)
        hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False, percentile=95)
        if self.config.co_classify:
            acc_metric = ConfusionMatrixMetric(include_background=True, metric_name=['accuracy', 'f1 score'], reduction="mean", get_not_nans=False)
        
        if self.config.optimizer == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        
        if self.config.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.config.T0, T_mult=self.config.Tmult, eta_min=1e-5, last_epoch=-1)
        elif self.config.lr_scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
        else:
            scheduler = None
        
        if self.config.is_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        if self.config.co_classify:
            best_metric = (-1, -1, 999, -1, -1)
        else:
            best_metric = (-1, -1, 999)
            
        best_metric_epoch = -1
        epoch_loss_values = []
        metric_values = []
        epoch_times = []
        total_start = time.time()
        patience = 0
        
        def cc(img1, img2):
            eps = torch.finfo(torch.float32).eps
            """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
            # N, C, _, _ = img1.shape
            N, C, _ = img1.shape
            img1 = img1.reshape(N, C, -1)
            img2 = img2.reshape(N, C, -1)
            img1 = img1 - img1.mean(dim=-1, keepdim=True)
            img2 = img2 - img2.mean(dim=-1, keepdim=True)
            cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1**2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
            cc = torch.clamp(cc, -1., 1.)
            return cc.mean()
        
        def calc_DiversityLoss(atten):
            N = atten.size(1)
            max_values, _ = atten.max(dim=1, keepdim=True)  # max over the n dimension
            div_att = max_values.sum(dim=2) / N  # mean over the hw dimension
            diversity_loss = 1 - div_att
            return diversity_loss.mean()
        
        # def calculate_ccloss(df, bf):
        #     ccloss = 0
        #     for i in range(len(df) - 1):
        #         for j in range(i+1, len(df)):
        #             # cc(df[i], df[j]) ** 2 -> [0, 1]
        #             # cc(bf[i], bf[j]) ** 2 - 1 -> [-1, 0]
        #             ccloss += cc(df[i], df[j]) ** 2 / (cc(bf[i], bf[j]) + 1.01)
        #             # ccloss += cc(df[i], df[j]) ** 2 - (cc(bf[i], bf[j]) ** 2 - 1)
        #     return ccloss
        def calculate_ccloss(df, bf):
            ccloss = 0
            for i in range(len(bf) - 1):
                for j in range(i+1, len(bf)):
                    # cc(df[i], df[j]) ** 2 -> [0, 1]
                    # cc(bf[i], bf[j]) + 1 + 1e-7 -> [0, 2]
                    # ccloss += cc(df[i], df[j]) ** 2 / (1 - cc(bf[i], bf[j]) ** 2)
                    ccloss += cc(bf[i], bf[j]) ** 2
                    # ccloss += cc(df[i], df[j]) ** 2 / (cc(bf[i], bf[j]) + 1 + 1e-7)
                    # ccloss += cc(df[i], df[j]) ** 2 - (cc(bf[i], bf[j]) ** 2 - 1)
            ccloss /= (len(bf) * (len(bf) - 1) / 2)
            return ccloss
        
        for epoch in range(self.config.max_epochs):
            epoch_start = time.time()
            self.logger.log("-" * 10)
            self.logger.log(f"epoch {epoch + 1}/{self.config.max_epochs}")

            self.model.train()
            epoch_loss = 0
            train_loader_iterator = iter(self.train_loader)
            
            patient_metrics = {}

            # using step instead of iterate through train_loader directly to track data loading time
            # steps are 1-indexed for printing and calculation purposes
            for step in range(1, len(self.train_loader) + 1):
                step_start = time.time()
                batch_data = next(train_loader_iterator)
                if self.config.multi_view:
                    input = []
                    label = []
                    view = []
                    
                    if self.config.is_video:
                        if self.config.mv:
                            for i_view in range(len(self.config.views)):
                                input.append(batch_data[i_view]["image"].to(self.device).unsqueeze(2))
                                label.append(batch_data[i_view]["label"].to(self.device))
                                view.append(batch_data[i_view]["view"].to(self.device).unsqueeze(2))
                            inputs = rearrange(torch.stack(input, dim=1), 'b v t c h w -> (b v t) c h w')
                            labels = rearrange(torch.stack(label, dim=1), 'b v c h w -> (b v) c h w')
                            views = rearrange(torch.stack(view, dim=1), 'b v t c -> (b v t) c')
                        else:
                            inputs = rearrange(batch_data["image"].to(self.device).unsqueeze(2), 'b t c h w -> (b t) c h w')
                            labels = batch_data["label"].to(self.device)
                            views = batch_data["view"].to(self.device)
                    else:
                        for i_view in range(len(self.config.views)):
                            input.append(batch_data[i_view]["image"].to(self.device).unsqueeze(1))
                            label.append(batch_data[i_view]["label"].to(self.device).unsqueeze(1))
                            view.append(batch_data[i_view]["view"].to(self.device).unsqueeze(1))
                        inputs = rearrange(torch.cat(input, dim=1), 'b v c h w -> (b v) c h w')
                        labels = rearrange(torch.cat(label, dim=1), 'b v c h w -> (b v) c h w')
                        views = rearrange(torch.cat(view, dim=1).unsqueeze(-1), 'b v c -> (b v) c')
                else:
                    inputs, labels, views = (
                            batch_data["image"].to(self.device),
                            batch_data["label"].to(self.device),
                            batch_data["view"].to(self.device),
                        )
                    if self.config.is_video:
                        inputs = inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        labels = labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                        views = views.view(-1, 1)

                # plot_3d_mat(inputs.to(dtype=torch.float32), epoch, self.save_path, name='image')
                # plot_3d_mat(labels.to(dtype=torch.float32), step, self.save_path, name='label')
                optimizer.zero_grad()

                if self.config.is_amp:
                    with torch.cuda.amp.autocast():
                        if self.config.co_classify:
                            outputs, pred_views = self.model(inputs)
                            loss = self.calculate_loss(outputs, labels, pred_views, views)
                        else:
                            outputs = self.model(inputs)
                            # outputs = rearrange(outputs, '(b v t) c h w -> t (b v) c h w', v=3, t=7)[0]
                            disloss = 0
                            if type(outputs) == tuple:
                                outputs, atten = outputs[0], outputs[1]
                                if self.config.divloss:
                                    disloss = calc_DiversityLoss(atten)
                            # for f in bf:
                            #     ccloss += calculate_ccloss(df, f)
                            loss = self.calculate_loss(outputs, labels)
                            if self.config.divloss:
                                loss += disloss
                else:
                    if self.config.co_classify:
                        outputs, pred_views = self.model(inputs)
                        loss = self.calculate_loss(outputs, labels, pred_views, views)
                    else:
                        outputs = self.model(inputs)
                        # ccloss = calculate_ccloss(df, bf)
                        loss = self.calculate_loss(outputs, labels)
                        # loss += 2*ccloss
                        
                # plot_3d_mat(torch.argmax(outputs, dim=1, keepdim=True).to(dtype=torch.float32), epoch, self.save_path, name='pred')
                
                if self.config.is_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_len = math.ceil(self.len_train_ds / self.train_loader.batch_size)
                if step % 50 == 1:
                    self.logger.log(
                        f"{step}/{epoch_len}, train_loss: {loss.item():.4f}" f" step time: {(time.time() - step_start):.4f}"
                    )
            if scheduler is not None:
                scheduler.step()
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            if not self.debug:
                wandb.log({"train_loss": epoch_loss})
            self.logger.log(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % self.config.val_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    val_loader_iterator = iter(self.val_loader)

                    for _ in range(len(self.val_loader)):
                        val_data = next(val_loader_iterator)
                        if self.config.multi_view:
                            input = []
                            label = []
                            view = []
                            name = []
                            
                            if self.config.is_video:
                                if self.config.mv:
                                    for i_view in range(len(self.config.views)):
                                        input.append(val_data[i_view]["image"].to(self.device).unsqueeze(2))
                                        label.append(val_data[i_view]["label"].to(self.device))
                                        view.append(val_data[i_view]["view"].to(self.device).unsqueeze(2))
                                        name.append(val_data[i_view]["name"])
                                    val_inputs = rearrange(torch.stack(input, dim=1), 'b v t c h w -> (b v t) c h w')
                                    val_labels = rearrange(torch.stack(label, dim=1), 'b v c h w -> (b v) c h w')
                                    val_views = rearrange(torch.stack(view, dim=1), 'b v t c -> (b v t) c')
                                else:
                                    val_inputs = rearrange(val_data["image"].to(self.device).unsqueeze(
                                        2), 'b t c h w -> (b t) c h w')
                                    val_labels = val_data["label"].to(self.device)
                                    val_views = val_data["view"].to(self.device)
                                    name = [val_data["name"]]
                            else:
                                for i_view in range(len(self.config.views)):
                                    input.append(val_data[i_view]["image"].to(self.device).unsqueeze(1))
                                    label.append(val_data[i_view]["label"].to(self.device).unsqueeze(1))
                                    view.append(val_data[i_view]["view"].to(self.device).unsqueeze(1))
                                    name.append(val_data[i_view]["name"])
                                val_inputs = rearrange(torch.cat(input, dim=1), 'b v c h w -> (b v) c h w')
                                val_labels = rearrange(torch.cat(label, dim=1), 'b v c h w -> (b v) c h w')
                                val_views = rearrange(torch.cat(view, dim=1).unsqueeze(-1), 'b v c -> (b v) c')
                        else:
                            val_inputs, val_labels, val_views = (
                                val_data["image"].to(self.device),
                                val_data["label"].to(self.device),
                                val_data["view"].to(self.device),
                            )
                        
                            if self.config.is_video:
                                val_inputs = val_inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                                val_labels = val_labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                                val_views = val_views.view(-1, 1)
                        
                        if self.config.is_amp:
                            with torch.cuda.amp.autocast():
                                if self.config.co_classify:
                                    val_outputs, val_pred_views = self.model(val_inputs)
                                else:
                                    val_outputs = self.model(val_inputs)
                                    # val_outputs = rearrange(val_outputs, '(b v t) c h w -> t (b v) c h w', v=3, t=7)[0]
                                    if type(val_outputs) == tuple:
                                        val_outputs, atten = val_outputs[0], val_outputs[1]
                                    
                                if self.config.multi_output:
                                    val_outputs = val_outputs[-1]

                                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                                if self.config.co_classify:
                                    val_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(val_pred_views.squeeze(-1).squeeze(-1))]
                                    val_views = [post_view(i).to(self.device) for i in decollate_batch(val_views)]
                                
                                dice_ = dice_metric(y_pred=val_outputs, y=val_labels)
                                nsd_ = nsd_metric(y_pred=val_outputs, y=val_labels)
                                hd95_ = hd95_metric(y_pred=val_outputs, y=val_labels)
                                
                                for i in range(len(name)):
                                    if name[i][0].split("/")[2] not in patient_metrics:
                                        patient_metrics[name[i][0].split("/")[2]] = {}
                                    if name[i][0].split("/")[4] not in patient_metrics[name[i][0].split("/")[2]]:
                                        patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]] = {}
                                    if name[i][0].split("/")[-1] in patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]].keys():
                                        old_dice, old_nsd, old_hd95 = patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split("/")[-1]]
                                        if old_dice < dice_[i].item():
                                            patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split("/")[-1]] = (dice_[i].item(), nsd_[i].item(), hd95_[i].item())
                                    else:
                                        patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split(
                                            "/")[-1]] = (dice_[i].item(), nsd_[i].item(), hd95_[i].item())
                                
                                if self.config.co_classify:
                                    acc_metric(y_pred=val_pred_views, y=val_views)
                            
                        else:
                            if self.config.co_classify:
                                val_outputs, val_pred_views = self.model(val_inputs)
                            else:
                                val_outputs = self.model(val_inputs)
                                
                            if self.config.multi_output:
                                val_outputs = val_outputs[-1]

                            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                            if self.config.co_classify:
                                val_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(val_pred_views.squeeze(-1).squeeze(-1))]
                                val_views = [post_view(i).to(self.device) for i in decollate_batch(val_views)]

                            dice_metric(y_pred=val_outputs, y=val_labels)
                            nsd_metric(y_pred=val_outputs, y=val_labels)
                            hd95_metric(y_pred=val_outputs, y=val_labels)
                            if self.config.co_classify:
                                acc_metric(y_pred=val_pred_views, y=val_views)

                    # print(patient_metrics)
                    metrics_dict = get_mean_and_std_metrics(patient_metrics)
                    new_dice = []
                    new_nsd = []
                    new_hd95 = []
                    for k, v in metrics_dict.items():
                        new_dice.append(v[0])
                        new_nsd.append(v[1])
                        new_hd95.append(v[2])
                    
                    dice = np.mean(new_dice)
                    # dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                    
                    nsd = np.mean(new_nsd)
                    # nsd = nsd_metric.aggregate().item()
                    nsd_metric.reset()
                    
                    hd95 = np.mean(new_hd95)
                    # hd95 = hd95_metric.aggregate().item()
                    hd95_metric.reset()
                    
                    if self.config.co_classify:
                        confusion = acc_metric.aggregate()
                        acc_metric.reset()
                        acc, f1 = confusion[0].item(), confusion[1].item()
                        metric_values.append((dice, nsd, hd95, acc, f1))
                    else:
                        metric_values.append((dice, nsd, hd95))
                    
                    if hd95 == 0:
                        hd95 = 999
                        
                    if not self.debug:
                        wandb.log({"val_dice": dice, "val_nsd": nsd, "val_hd95": hd95})
                    
                    if (dice + nsd - hd95 * 0.04) > (best_metric[0] + best_metric[1] - best_metric[2] * 0.04) and patience < self.config.early_stopping_patience:
                        if self.config.co_classify:
                            best_metric = (dice, nsd, hd95, acc, f1)
                        else:
                            best_metric = (dice, nsd, hd95)
                        best_metric_epoch = epoch + 1
                        torch.save(self.model.state_dict(), os.path.join(self.save_path, f'fold{fold_index}', f"{self.method_name}_best.pt"))
                        self.logger.log("saved new best metric model")
                        patience = 0
                    elif patience >= self.config.early_stopping_patience:
                        self.logger.log("early stopping patience reached")
                        break
                    else:
                        self.logger.log("current epoch patience is {}".format(patience))
                        patience += 1

                    self.logger.log(
                        f"current epoch: {epoch + 1} current"
                        f" mean dice: {dice:.4f}"
                        f" mean nsd: {nsd:.4f}"
                        f" mean hd95: {hd95:.4f}"
                        f" best mean dice: {best_metric[0]:.4f}"
                        f" best mean nsd: {best_metric[1]:.4f}"
                        f" best mean hd95: {best_metric[2]:.4f}"
                        f" at epoch: {best_metric_epoch}"
                    )
 
            self.logger.log(f"time consuming of epoch {epoch + 1} is:" f" {(time.time() - epoch_start):.4f}")
            epoch_times.append(time.time() - epoch_start)

        total_time = time.time() - total_start
        self.logger.log(
            f"train completed, best_metric: {best_metric[0]:.4f} {best_metric[1]:.4f} {best_metric[2]:.4f}"
            f" at epoch: {best_metric_epoch}"
            f" total time: {total_time:.4f}"
        )
        return (
            epoch + 1,
            epoch_loss_values,
            metric_values,
            epoch_times,
            total_time,
        )
    
    def test_process_mv(self, fold_index):
        self.model.to(self.device)
        
        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2), KeepLargestConnectedComponent(is_onehot=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])
        post_pred_view = Compose([AsDiscrete(argmax=True, to_onehot=len(self.config.views_dict[self.config.dataset]))])
        post_view = Compose([AsDiscrete(to_onehot=len(self.config.views_dict[self.config.dataset]))])

        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        nsd_metric = SurfaceDiceMetric(include_background=False, reduction="mean", get_not_nans=False, class_thresholds=self.config.class_thresholds)
        hd95_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False, percentile=95)
        if self.config.co_classify:
            acc_metric = ConfusionMatrixMetric(include_background=True, metric_name=['accuracy', 'f1 score'], reduction="mean", get_not_nans=False)
        
        min_dice = (2, -1)
        min_nsd = (2, -1)
        max_hd95 = (0, -1)
        plot_data = [None, None, None]
        total_start = time.time()
        self.model.eval()
        
        test_loader_iterator = iter(self.test_loader)
        
        best_dice = 0
        best_p = None
        
        patient_metrics = {}
        
        for idx in range(len(self.test_loader)):
            test_data = next(test_loader_iterator)
            if self.config.multi_view:
                input = []
                label = []
                view = []
                name = []
                if self.config.is_video:
                    if self.config.mv:
                        for i_view in range(len(self.config.views)):
                            input.append(test_data[i_view]["image"].to(self.device).unsqueeze(2))
                            label.append(test_data[i_view]["label"].to(self.device))
                            view.append(test_data[i_view]["view"].to(self.device).unsqueeze(2))
                            name.append(test_data[i_view]["name"])
                        test_inputs = rearrange(torch.stack(input, dim=1), 'b v t c h w -> (b v t) c h w')
                        test_labels = rearrange(torch.stack(label, dim=1), 'b v c h w -> (b v) c h w')
                        test_views = rearrange(torch.stack(view, dim=1), 'b v t c -> (b v t) c')
                    else:
                        test_inputs = rearrange(test_data["image"].to(self.device).unsqueeze(
                            2), 'b t c h w -> (b t) c h w')
                        test_labels = test_data["label"].to(self.device)
                        test_views = test_data["view"].to(self.device)
                        name = [test_data["name"]]
                else:
                    for i_view in range(len(self.config.views)):
                        input.append(test_data[i_view]["image"].to(self.device).unsqueeze(1))
                        label.append(test_data[i_view]["label"].to(self.device).unsqueeze(1))
                        view.append(test_data[i_view]["view"].to(self.device).unsqueeze(1))
                        name.append(test_data[i_view]["name"])

                    test_inputs = rearrange(torch.cat(input, dim=1), 'b v c h w -> (b v) c h w')
                    test_labels = rearrange(torch.cat(label, dim=1), 'b v c h w -> (b v) c h w')
                    test_views = rearrange(torch.cat(view, dim=1).unsqueeze(-1), 'b v c -> (b v) c')
            else:
                test_inputs, test_labels, test_views = (
                    test_data["image"].to(self.device),
                    test_data["label"].to(self.device),
                    test_data["view"].to(self.device),
                )
                
                if self.config.is_video:
                    test_inputs = test_inputs.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    test_labels = test_labels.view(-1, 1, self.config.image_size[0], self.config.image_size[1])
                    test_views = test_views.view(-1, 1)
            
            with torch.no_grad():
                # plot_3d_mat(F.sigmoid(rearrange(test_inputs, '(b v t) c h w -> b (v t c) w h', v=len(self.config.views), t=self.config.video_length)[0]), 0, 'test_png', name='input')
                # plot_3d_mat(F.sigmoid(rearrange(test_labels, '(b v) c h w -> b (v c) w h', v=len(self.config.views))[0]), 0, 'test_png', name='label')
                # exit()
                if self.config.is_amp:
                    with torch.cuda.amp.autocast():
                        if self.config.co_classify:
                            test_outputs, test_pred_views = self.model(test_inputs)
                        else:
                            test_outputs = self.model(test_inputs)
                            # test_outputs = rearrange(test_outputs, '(b v t) c h w -> t (b v) c h w', v=3, t=7)[0]
                            if type(test_outputs) == tuple:
                                test_outputs = test_outputs[0]
                            # test_outputs, atten = self.model(test_inputs)
                            # name_dict = {0: 'x0', 1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4', 5: 'u4', 6: 'u3', 7: 'u2', 8: 'u1'}
                            # for i in range(len(vis)):
                            #     for j in range(vis[i].shape[1]):
                            #         plot_3d_mat(vis[i][:, j, :, :].to(dtype=torch.float32), j, self.save_path, name=f'{name_dict[i]}')
                            # exit()
                else:
                    if self.config.co_classify:
                        test_outputs, test_pred_views = self.model(test_inputs)
                    else:
                        test_outputs = self.model(test_inputs)
                        
                if self.config.multi_output:
                    test_outputs = test_outputs[-1]
                
                test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
                test_labels = [post_label(i) for i in decollate_batch(test_labels)]

                # for i in range(len(test_outputs)):
                #     #if f'{name[i][0].split("image")[-1].split(".png")[0]}' == '0':
                #     plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join('test_pred_result', 
                #         f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[i][0].split("/")[2]}', f'{name[i][0].split("/")[4]}'), name=f'pred{name[i][0].split("image")[-1].split(".png")[0]}', cmap='gray')
 
                if self.config.co_classify:
                    test_pred_views = [post_pred_view(i).to(self.device) for i in decollate_batch(test_pred_views.squeeze(-1).squeeze(-1))]
                    test_views = [post_view(i).to(self.device) for i in decollate_batch(test_views)]
                    
                dice_ = dice_metric(y_pred=test_outputs, y=test_labels)
                nsd_ = nsd_metric(y_pred=test_outputs, y=test_labels)
                hd95_ = hd95_metric(y_pred=test_outputs, y=test_labels)
                for i in range(len(name)):
                    if name[i][0].split("/")[2] not in patient_metrics:
                        patient_metrics[name[i][0].split("/")[2]] = {}
                    if name[i][0].split("/")[4] not in patient_metrics[name[i][0].split("/")[2]]:
                        patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]] = {}
                    if name[i][0].split("/")[-1] in patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]].keys():
                        old_dice, old_nsd, old_hd95 = patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split("/")[-1]]
                        if old_dice < dice_[i].item():
                            patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split("/")[-1]] = (dice_[i].item(), nsd_[i].item(), hd95_[i].item())
                            plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join(f'methods_pred/{self.config.dataset}', f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[i][0].split("/")[2]}', f'{name[i][0].split("/")[4]}'), name=f'pred{name[i][0].split("label")[-1].split(".png")[0]}', cmap='gray')
                    else:
                        patient_metrics[name[i][0].split("/")[2]][name[i][0].split("/")[4]][name[i][0].split("/")[-1]] = (dice_[i].item(), nsd_[i].item(), hd95_[i].item())
                        plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join(f'methods_pred/{self.config.dataset}', f'{self.config.project_name}-pred', f'fold{fold_index}', f'{name[i][0].split("/")[2]}', f'{name[i][0].split("/")[4]}'), name=f'pred{name[i][0].split("label")[-1].split(".png")[0]}', cmap='gray')
                
                if self.config.co_classify:
                    confusion_ = acc_metric(y_pred=test_pred_views, y=test_views)

                # if dice_.mean() > best_dice:
                #     best_dice = dice_.mean()
                #     best_p = name
                
                if not self.config.is_video:
                    if dice_.min() < min_dice[0]:
                        min_dice = (dice_.min(), idx)
                        # plot_data[0] = (test_outputs[0], test_labels[0], test_inputs)
                        # plot_3d_mat(torch.argmax(test_outputs[0], dim=0, keepdim=True).to(dtype=torch.float32), 0, '.', name='pred', cmap='gray')
                    if nsd_.min() < min_nsd[0]:
                        min_nsd = (nsd_.min(), idx)
                        # plot_data[1] = (test_outputs[0], test_labels[0], test_inputs)
                    if hd95_.max() > max_hd95[0]:
                        max_hd95 = (hd95_.max(), idx)
                        # plot_data[2] = (test_outputs[0], test_labels[0], test_inputs)

        # print(best_dice, best_p)
        # print(patient_metrics)

        # # 创建一个空的列表来存储行数据
        # rows = []

        # # 将数据字典转换为列表
        # for patient_id, views in patient_metrics.items():
        #     for view, frames in views.items():
        #         for frame, metrics in frames.items():
        #             dice, nsd, hd95 = metrics
        #             rows.append({
        #                 'Patient ID': patient_id,
        #                 'View': view,
        #                 'Frame': frame,
        #                 'Dice': dice,
        #                 'NSD': nsd,
        #                 'HD95': hd95
        #             })

        # # 将列表转换为DataFrame
        # df = pd.DataFrame(rows, columns=['Patient ID',
        #                 'View', 'Frame', 'Dice', 'NSD', 'HD95'])

        # # 保存DataFrame到Excel文件
        # df.to_excel('patient_data_ext.xlsx', index=False)
        # exit()
        
        metrics_dict = get_mean_and_std_metrics(patient_metrics)
        new_dice = []
        new_nsd = []
        new_hd95 = []
        for k, v in metrics_dict.items():
            new_dice.append(v[0])
            new_nsd.append(v[1])
            new_hd95.append(v[2])
            
        # dice = dice_metric.aggregate().item()
        dice_metric.reset()
        
        # nsd = nsd_metric.aggregate().item()
        nsd_metric.reset()
        
        # hd95 = hd95_metric.aggregate().item()
        hd95_metric.reset()
        
        if self.config.co_classify:
            confusion = acc_metric.aggregate()
            acc_metric.reset()
            acc, f1 = confusion[0].item(), confusion[1].item()
        
        total_time = time.time() - total_start
        
        if self.config.co_classify:
            self.logger.log(
                f" mean dice: {np.mean(new_dice):.4f}"
                f" mean nsd: {np.mean(new_nsd):.4f}"
                f" mean hd95: {np.mean(new_hd95):.4f}"
                f" mean acc: {acc:.4f}"
                f" mean f1: {f1:.4f}"
                f" total time: {total_time:.4f}"
            )
        else:
            self.logger.log(
                f" mean dice: {np.mean(new_dice):.4f}±{np.std(new_dice):.4f}"
                f" mean nsd: {np.mean(new_nsd):.4f}±{np.std(new_nsd):.4f}"
                f" mean hd95: {np.mean(new_hd95):.4f}±{np.std(new_hd95):.4f}"
                f" total time: {total_time:.4f}"
            )
        if not self.config.is_video:
            self.logger.log(
                f" min dice: {min_dice[0]:.4f} in {min_dice[1]}th batch |"
                f" min nsd: {min_nsd[0]:.4f} in {min_nsd[1]}th batch |"
                f" max hd95: {max_hd95[0]:.4f} in {max_hd95[1]}th batch"
            )
        
        test_metric = (np.mean(new_dice), np.mean(new_nsd), np.mean(new_hd95), np.std(new_dice), np.std(new_nsd), np.std(new_hd95))
        # if not self.debug:
        #     wandb.log({"test_dice": dice, "test_nsd": nsd, "test_hd95": hd95})
            
        # if not self.config.is_video:
        #     np_plot_data = []
        #     for plot in plot_data:
        #         a = plot[0].argmax(dim=0).squeeze().transpose(0, 1).cpu().numpy()
        #         b = plot[1].argmax(dim=0).squeeze().transpose(0, 1).cpu().numpy()
        #         c = plot[2].squeeze().transpose(0, 1).cpu().numpy()
        #         a = resample(a, mode='nearest')
        #         b = resample(b, mode='nearest')
        #         c = resample(c, mode='linear')
        #         np_plot_data.append((a, b, c))
            
        #     plot_example(fold_save_path, np_plot_data, min_dice, min_nsd, max_hd95)
        
        return test_metric
