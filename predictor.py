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


class Predictor():
    def __init__(self,
                 method_name=None,
                 model=None,
                 predict_path={'HCM': 'HCM/test_dataset.pkl'},
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
            
        self.predict_path_dict = predict_path

    def transformations(self, device="cuda:0"):
        predict_transforms = [
            LoadImaged(keys=["image"], image_only=False),
            # EnsureChannelFirstd(keys=["image"]),
            # CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(keys=["image"], spatial_size=self.config.image_size, mode=(
                "bilinear")),
            # Spacingd(keys=["image", "label"], pixdim=(1.35, 1), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
            # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
            EnsureTyped(keys=["image", "view"],
                        device=device, track_meta=True)
        ]

        trans = Compose(predict_transforms).set_random_state(seed=0)

        return trans

    def make_multi_views_predict_dataset(self, predict_path='HCM/test_dataset.pkl', es_info_path='HCM/info.csv'):
        predict_images = pickle.load(open(predict_path, 'rb'))
        patients = list(predict_images.keys())
        if 'CAMUS' in predict_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_camus_info
        elif 'HCM' in predict_path:
            info = pd.read_csv(f'{self.config.dataset}/info.csv')
            read_info_func = read_hcm_info
        if self.config.frame_align == 'RA' or 'HMCQU' in predict_path:
            frame_align_func = frame_align
        elif self.config.frame_align == 'FA' or 'CAMUS' in predict_path:
            frame_align_func = frame_align_Full
        elif self.config.frame_align == 'ES':
            frame_align_func = frame_align_EDES

        if self.config.is_video:
            predict_image_temps = {self.config.views_dict[self.config.dataset][i]: [] for i in self.config.views}
            for patient in patients:
                flag = True
                frames = {}
                for view in predict_images[patient]:
                    if len(predict_images[patient][view]) == 0:
                        flag = False
                        break
                    else:
                        frames[view] = predict_images[patient][view]
                if flag:
                    if self.config.mv:
                        if 'CAMUS' in predict_path:
                            ed, es = read_info_func(info, patient.split('/')[-1])
                            frames = frame_align_Full(frames, ed, None, es)
                        elif 'HCM' in predict_path:
                            es = read_info_func(info, patient.split('/')[-1])
                            start = {k: 0 for k in frames.keys()}
                            es = {k: es[idx] for idx, k in enumerate(frames.keys())}
                            end = {k: len(frames[k]) for k in frames.keys()}
                            frames = preprocessing(frames, start, es, end)
                        else:
                            frames = frame_align_func(frames, None, None, None)

                    for view in frames:
                        if self.config.views_dict[self.config.dataset].index(view) not in self.config.views:
                            continue
                        predict_image_temps[view].extend(frames[view])
        
        predict_dss = []
        for view in predict_image_temps.keys():
            predict_images = predict_image_temps[view]

            predict_dicts = [{"image": image_name,
                           "view": [self.config.views_dict[self.config.dataset].index(view)]*len(image_name)
                           if self.config.is_video else self.config.views_dict[self.config.dataset].index(view),
                              "name": image_name} for image_name in predict_images]

            predict_trans = self.transformations(device=self.device)

            predict_ds = CacheDataset(data=predict_dicts, transform=predict_trans,
                                   cache_rate=1.0, num_workers=5, copy_cache=False)
            predict_dss.append(predict_ds)

        if self.config.mv:
            predict_ds = ZipDataset(predict_dss)

        return predict_ds

    def prediction(self):
        set_track_meta(False)
        if not self.debug:
            wandb.init(project=self.method_name, config=self.config)
        self.save_path = os.path.join(self.config.pred_save_path, self.config.dataset)
        print(self.predict_path_dict, self.config.dataset)
        self.predict_path = self.predict_path_dict[self.config.dataset]
        model_save_path = os.path.join(self.save_path)
        monai_start = time.time()
        '''
            -------------predict--------------
        '''
        if self.config.multi_view:
            predict_ds = self.make_multi_views_predict_dataset(self.predict_path)

        self.predict_loader = ThreadDataLoader(predict_ds, num_workers=0, batch_size=1)

        self.len_predict_ds = len(predict_ds)
        self.model.load_state_dict(torch.load(os.path.join(model_save_path, f"{self.method_name}_best.pt")))
        self.predict_process_mv()
        # dice, nsd, hd95 = self.test_process_plot()
        wandb.finish()
        # exit()

    def predict_process_mv(self):
        self.model.to(self.device)
        total_start = time.time()
        self.model.eval()

        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2),
                            KeepLargestConnectedComponent(is_onehot=True)])

        predict_loader_iterator = iter(self.predict_loader)

        for idx in range(len(self.predict_loader)):
            predict_data = next(predict_loader_iterator)
            if self.config.multi_view:
                input = []
                view = []
                name = []
                if self.config.is_video:
                    if self.config.mv:
                        for i_view in range(len(self.config.views)):
                            input.append(predict_data[i_view]["image"].to(
                                self.device).unsqueeze(2))
                            view.append(predict_data[i_view]["view"].to(
                                self.device).unsqueeze(2))
                            name.append(predict_data[i_view]["name"])
                        predict_inputs = rearrange(torch.stack(
                            input, dim=1), 'b v t c h w -> (b v t) c h w')
                        predict_views = rearrange(torch.stack(
                            view, dim=1), 'b v t c -> (b v t) c')
                    else:
                        predict_inputs = rearrange(predict_data["image"].to(self.device).unsqueeze(
                            2), 'b t c h w -> (b t) c h w')
                        predict_views = predict_data["view"].to(self.device)
                        name = [predict_data["name"]]
                else:
                    for i_view in range(len(self.config.views)):
                        input.append(predict_data[i_view]["image"].to(
                            self.device).unsqueeze(1))
                        view.append(predict_data[i_view]["view"].to(
                            self.device).unsqueeze(1))
                        name.append(predict_data[i_view]["name"])

                    predict_inputs = rearrange(
                        torch.cat(input, dim=1), 'b v c h w -> (b v) c h w')
                    predict_views = rearrange(
                        torch.cat(view, dim=1).unsqueeze(-1), 'b v c -> (b v) c')
            else:
                predict_inputs, predict_views = (
                    predict_data["image"].to(self.device),
                    predict_data["view"].to(self.device),
                )

                if self.config.is_video:
                    predict_inputs = predict_inputs.view(
                        -1, 1, self.config.image_size[0], self.config.image_size[1])
                    predict_views = predict_views.view(-1, 1)

            with torch.no_grad():
                if self.config.is_amp:
                    with torch.cuda.amp.autocast():
                        predict_outputs = self.model(predict_inputs)
                        if type(predict_outputs) == tuple:
                            predict_outputs = predict_outputs[0]
                else:
                    predict_outputs = self.model(predict_inputs)

                if self.config.multi_output:
                    predict_outputs = predict_outputs[-1]

                test_outputs = [post_pred(i)
                                for i in decollate_batch(predict_outputs)]

                for i in range(len(name)):
                    plot_test_result(test_outputs[i].to(dtype=torch.float32).transpose(-1, -2).argmax(dim=0, keepdim=True), os.path.join(
                        f'prediction', f'{name[i][0][0].split("/")[2]}', f'{name[i][0][0].split("/")[4]}'), name=f'pred{name[i][0][0].split("image")[-1].split(".png")[0]}', cmap='gray')
