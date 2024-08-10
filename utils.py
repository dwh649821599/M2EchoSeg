import os
import logging
from typing import Any
import matplotlib.pyplot as plt
from monai.visualize.utils import matshow3d
import sys
import numpy as np
import torch
import random
import cv2
from PIL import Image
import pandas as pd
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')

class TrainConfig:
    def __init__(self, project_name='unet-train', 
                    max_epochs=100, 
                    batch_size=32, 
                    early_stopping_patience=20, 
                    learning_rate=5e-2,
                    val_interval=1,
                    is_cv=False,
                    auto_edes=False,
                    cv_folds=5,
                    image_size=(192, 192),
                    views=[0, 1, 2],
                    dataset='HCM',
                    save_path='result',
                    pred_save_path='model',
                 views_dict={'HCM': ['A2C1', 'A3C1', 'A4C1'], 'CAMUS': ['A2C', 'A4C'], 'HCM-ext': ['A2C1', 'A3C1', 'A4C1']},
                    co_classify=False,
                    is_video=False,
                    multi_view=False,
                    n_segment=12,
                    n_overlap=2,
                    video_length=7,
                    class_thresholds=[3],
                    multi_output=False,
                    ssim_loss=False,
                    is_amp=True,
                    dropout=0.1,
                    optimizer='SGD',
                    momentum=0.9,
                    weight_decay=1e-7,
                    activation='elu',
                    features=[16, 32, 64, 128, 256, 16],
                    lr_scheduler='CosineAnnealingLR',
                    T0=15,
                    Tmult=3,
                    mv=False,
                    NUM_BATCHES_TO_LOG=100,
                    frame_align = 'ES',
                    ext_test=False,
                    divloss=True
                    ):
        self.project_name = project_name
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.val_interval = val_interval
        self.auto_edes = auto_edes
        self.is_cv = is_cv
        self.mv = mv
        self.cross_validation_folds = cv_folds if is_cv else 1
        self.views = views
        self.dataset = dataset
        self.save_path = save_path
        self.pred_save_path = pred_save_path
        self.co_classify = co_classify
        self.is_video = is_video
        self.multi_view = multi_view
        self.video_length = video_length if is_video else 1
        self.n_segment = n_segment if is_video else 1
        self.n_overlap = n_overlap if is_video else 0
        self.multi_output = multi_output
        self.class_thresholds=class_thresholds
        self.views_dict = views_dict
        self.ssim_loss = ssim_loss
        self.is_amp = is_amp
        self.dropout = dropout
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.activation = activation
        self.features = features
        self.lr_scheduler = lr_scheduler
        self.T0 = T0
        self.Tmult = Tmult
        self.NUM_BATCHES_TO_LOG = NUM_BATCHES_TO_LOG
        self.frame_align = frame_align
        self.ext_test = ext_test
        self.divloss = divloss
    
    def update(self, pdict):
        for k, v in pdict.items():
            setattr(self, k, v)
    
    def __str__(self) -> str:
        return str(self.__dict__)
    
    
def slide_window(num_frames, window_size=24, overlap=4):
    start = 0
    end = window_size
    windows = []
    while end <= num_frames:
        windows.append((start, end))
        start += window_size - overlap
        end += window_size - overlap
    if end > num_frames:
        windows.append((num_frames - window_size, num_frames))
    return windows

def resample(image, mode='linear'):
    if mode == 'linear':
        image = cv2.resize(image, (648, 480), interpolation=cv2.INTER_LINEAR)
    elif mode == 'nearest':
        image = cv2.resize(image, (648, 480), interpolation=cv2.INTER_NEAREST)
    return image


def plot_3d_mat(image, epoch, save_path, name='image', title='', cmap='coolwarm'):
    fig = plt.figure()
    matshow3d(image, fig=fig, title=title, cmap=cmap)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f"{name}_{epoch}.png"))
    plt.close()

def plot_test_result(image, save_path, name='image', cmap='gray'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = (image[0].cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), image)

def vis_feature_maps(image, save_path, name='image', title='', mode='mean'):
    """_summary_

    Args:
        image (_type_): B, C, H, W
        save_path (_type_): _description_
        name (str, optional): _description_. Defaults to 'image'.
        title (str, optional): _description_. Defaults to ''.
    """
    fig = plt.figure()
    if mode == 'mean':
        image = image.mean(1)
    elif type(mode) == int:
        image = image[:, mode]
    matshow3d(image, fig=fig, title=title)
    if not os.path.exists(os.path.join(save_path, f"visualization")):
        os.makedirs(os.path.join(save_path, f"visualization"))
    plt.savefig(os.path.join(save_path, f"visualization", f"{name}.png"))
    plt.close()

def frame_align(frames_dict, a, b, c):
    views = list(frames_dict.keys())
    max_len = max([len(frames_dict[i]) for i in views])

    for i in views:
        cur_len = len(frames_dict[i])
        while cur_len < max_len:
            random_index = np.random.randint(1, len(frames_dict[i]))
            frames_dict[i].insert(random_index, frames_dict[i][random_index])
            cur_len = len(frames_dict[i])

    return frames_dict

def read_hcm_info(info, patient_id):
    df = info
    es_dict = {}
    patient_id = str(patient_id)
    for _, row in df.iterrows():
        patient = row['patient']
        view = row['view']
        es = row['es']

        # Populate es_dict
        if patient not in es_dict:
            es_dict[patient] = {}
        es_dict[patient][view] = es

    for i in es_dict.keys():
        if str(i) == patient_id:
            patient_es_dict = es_dict[i]
            return patient_es_dict


def read_camus_info(info, patient_id):
    df = info
    # Initialize dictionaries
    ed_dict = {}
    es_dict = {}

    # Iterate over the DataFrame rows
    for _, row in df.iterrows():
        patient = row['patient']
        view = row['view']
        ed = row['ed']
        es = row['es']

        # Populate ed_dict
        if patient not in ed_dict:
            ed_dict[patient] = {}
        ed_dict[patient][view] = ed

        # Populate es_dict
        if patient not in es_dict:
            es_dict[patient] = {}
        es_dict[patient][view] = es

    patient_ed_dict = ed_dict.get(patient_id, {})
    patient_es_dict = es_dict.get(patient_id, {})
    
    return patient_ed_dict, patient_es_dict

def frame_align_Full(frames_dict, start_dict, mid_dict, end_dict):
    views = list(frames_dict.keys())
    trimmed_frames_dict = {}
    if mid_dict is not None:
        trimmed_mid_dict = {}
    max_length = max([abs(end_dict[i] - start_dict[i])
                     for i in range(len(views))]) + 1
    # Step 1: Trim the videos
    for i, view in enumerate(frames_dict.keys()):
        start, end = start_dict[i], end_dict[i]
        if start <= end:
            trimmed_frames = frames_dict[view][start:end + 1]
            if mid_dict is not None:
                new_mid = mid_dict[i] - start
        else:
            # Reverse if start is greater than end
            trimmed_frames = frames_dict[view][end:start + 1][::-1]
            if mid_dict is not None:
                new_mid = start - mid_dict[i]
        trimmed_frames_dict[view] = trimmed_frames
        if mid_dict is not None:
            trimmed_mid_dict[i] = new_mid

    # Step 2: Align the videos
    aligned_frames_dict = {}
    for i, view in enumerate(trimmed_frames_dict.keys()):
        frames = trimmed_frames_dict[view]
        if mid_dict is None:
            # If no mid_dict is provided, uniformly insert frames to match the max_length
            while len(frames) < max_length:
                insert_pos = random.randint(1, len(frames) - 1)
                frame_to_duplicate = frames[insert_pos]
                frames.insert(insert_pos, frame_to_duplicate)
        else:
            # Align the mid frames
            cur_len = len(frames)
            mid = trimmed_mid_dict[i]
            flag = True
            while cur_len < max_length:
                if flag:
                    insert_pos = random.randint(mid + 1, cur_len - 1)
                    flag = False
                else:
                    insert_pos = random.randint(0, mid - 1)
                    mid += 1
                    flag = True
                frame_to_duplicate = frames[insert_pos]
                frames.insert(insert_pos, frame_to_duplicate)
                cur_len = len(frames)
        aligned_frames_dict[view] = frames

    return aligned_frames_dict

def frame_align_EDES(frames_dict, start_dict, mid_dict, end_dict):
    views = list(frames_dict.keys())
    trimmed_frames_dict = {}
    if mid_dict is not None:
        trimmed_mid_dict = {}
        max_mid = max([abs(mid_dict[i] - start_dict[i])
                       for i in range(len(views))])
    max_length = max([abs(end_dict[i] - start_dict[i])
                     for i in range(len(views))]) + 1
    
    # Step 1: Trim the videos
    for i, view in enumerate(frames_dict.keys()):
        start, end = start_dict[i], end_dict[i]
        if start <= end:
            trimmed_frames = frames_dict[view][start:end + 1]
            if mid_dict is not None:
                new_mid = mid_dict[i] - start
        else:
            # Reverse if start is greater than end
            trimmed_frames = frames_dict[view][end:start + 1][::-1]
            if mid_dict is not None:
                new_mid = start - mid_dict[i]
        trimmed_frames_dict[view] = trimmed_frames
        if mid_dict is not None:
            trimmed_mid_dict[i] = new_mid

    # Step 2: Align the videos
    aligned_frames_dict = {}
    temp_frames_dict = {}

    for i, view in enumerate(trimmed_frames_dict.keys()):
        frames = trimmed_frames_dict[view]
        if mid_dict is None:
            # If no mid_dict is provided, uniformly insert frames to match the max_length
            while len(frames) < max_length:
                insert_pos = random.randint(1, len(frames) - 1)
                frame_to_duplicate = frames[insert_pos]
                frames.insert(insert_pos, frame_to_duplicate)
        else:
            # Align the mid frames
            mid = trimmed_mid_dict[i]
            while mid < max_mid:
                insert_pos = random.randint(0, mid - 1)
                mid += 1
                frame_to_duplicate = frames[insert_pos]
                frames.insert(insert_pos, frame_to_duplicate)
            temp_frames_dict[view] = frames

    if mid_dict is not None:
        max_length = max([len(temp_frames_dict[i]) for i in views])
        for i, view in enumerate(temp_frames_dict.keys()):
            frames = temp_frames_dict[view]
            cur_len = len(frames)
            while cur_len < max_length:
                insert_pos = random.randint(mid + 1, len(frames) - 1)
                frame_to_duplicate = frames[insert_pos]
                frames.insert(insert_pos, frame_to_duplicate)
                cur_len = len(frames)
            aligned_frames_dict[view] = frames
    return aligned_frames_dict()
    

def CrossViewConsistencyFrameMapping(frames_dict, start_dict, mid_dict, end_dict):
    # ED->ES
    ED2ES = [mid_dict[i] - start_dict[i] for i in frames_dict.keys()]
    max_len_EDES_view = list(frames_dict.keys())[np.argmax(ED2ES)]
    max_len_EDES = mid_dict[max_len_EDES_view] - \
        start_dict[max_len_EDES_view] + 1
    consistency_dict_EDES = {k: np.linspace(
        start_dict[k], mid_dict[k], num=max_len_EDES, dtype=np.int32) for k in frames_dict.keys()}
    new_ES_dict = {
        k: len(consistency_dict_EDES[k]) - 1 for k in frames_dict.keys()}
    # index_EDES = {k: np.searchsorted(consistency_dict_EDES[k], np.arange(mid_dict[k] - start_dict[k] + 1), side='left') for k in frames_dict.keys()}
    # mapping_EDES = {k: {kk: consistency_dict_EDES[kk][index_EDES[k]] for kk in list(set(frames_dict.keys()) - set(k))} for k in frames_dict.keys()}
    # print(consistency_dict_EDES)
    # ES->ED
    ES2ED = [end_dict[i] - mid_dict[i] - 1 for i in frames_dict.keys()]
    max_len_ESED_view = list(frames_dict.keys())[np.argmax(ES2ED)]
    max_len_ESED = end_dict[max_len_ESED_view] - \
        mid_dict[max_len_ESED_view] - 1
    consistency_dict_ESED = {k: np.linspace(
        mid_dict[k] + 1, end_dict[k] - 1, num=max_len_ESED, dtype=np.int32) for k in frames_dict.keys()}
    # index_ESED = {k: np.searchsorted(consistency_dict_ESED[k], np.arange(mid_dict[k] + 1, end_dict[k]), side='left') for k in frames_dict.keys()}
    # mapping_ESED = {k: {kk: consistency_dict_ESED[kk][index_ESED[k]] for kk in list(set(frames_dict.keys()) - set(k))} for k in frames_dict.keys()}
    # print(consistency_dict_ESED)
    # ED->ED
    # mapping = {k: {kk: np.concatenate([mapping_EDES[k][kk], mapping_ESED[k][kk]]) for kk in list(set(frames_dict.keys()) - set(k))} for k in frames_dict.keys()}
    # print(mapping)
    consistency_dict = {k: np.concatenate(
        [consistency_dict_EDES[k], consistency_dict_ESED[k]]) for k in frames_dict.keys()}
    new_ED_dict = {k: len(consistency_dict[k]) for k in frames_dict.keys()}
    # print(consistency_dict)
    return consistency_dict, new_ES_dict, new_ED_dict


def SymmetricSemanticReferenceSequenceSampling(frames, index, start, mid, end):
    len_EDES = mid - start
    len_ESED = end - mid - 1
    if len_ESED > len_EDES:
        consistency = np.linspace(
            start, mid, len_ESED, dtype=np.int32, endpoint=False)
        consistency = np.concatenate([consistency, np.arange(mid, end)])
        index = index[consistency]
    elif len_EDES > len_ESED:
        consistency = np.linspace(
            mid+1, end, len_EDES, dtype=np.int32, endpoint=False)
        consistency = np.concatenate([np.arange(start, mid+1), consistency])
        index = index[consistency]
    new_frames = []
    for i in range(len(index)):
        if i == len(index) // 2:
            # seq = [index[i], index[i-2], index[i-1], index[i+1]]
            # seq = [index[i], index[i-3], index[i-2], index[i-1], index[i+1], index[i+2], index[i+3]]
            # seq = [index[i], index[i]-2, index[i]-1, index[i]+2, index[i]+1]
            # rs7
            seq = [index[i], index[i-3], index[i-2], index[i-1], index[i+1], index[i+2], index[i+3]]
            # rs5
            # seq = [index[i], index[i-2], index[i-1], index[i+1], index[i+2]]
            # rs9
            # seq = [index[i], index[i-4], index[i-3], index[i-2], index[i-1], index[i+1], index[i+2], index[i+3], index[i+4]]
            # rs15
            # seq = [index[i], index[i-7], index[i-6], index[i-5], index[i-4], index[i-3], index[i-2],
            #        index[i-1], index[i+1], index[i+2], index[i+3], index[i+4], index[i+5], index[i+6], index[i+7]]
        else:
            # rs7
            seq = [index[i], index[len(index) // 2 * 2 - i - 1], index[i-2], index[i-1], 
                   index[(i+1) % len(index)], index[(i+2) % len(index)], index[(len(index) // 2 * 2 - i + 1) % len(index)]]
            # rs5
            # seq = [index[i], index[len(index) // 2 * 2 - i - 1], index[i-1],
            #         index[(i+1) % len(index)], index[(len(index) // 2 * 2 - i + 1) % len(index)]]
            # rs9
            # seq = [index[i], index[len(index) // 2 * 2 - i - 2], index[len(index) // 2 * 2 - i - 1], index[i-2], index[i-1],
            #        index[(i+1) % len(index)], index[(i+2) % len(index)], index[(len(index) // 2 * 2 - i + 1) % len(index)], index[(len(index) // 2 * 2 - i + 2) % len(index)]]
            # rs15
            # seq = [index[i], index[len(index) // 2 * 2 - i - 5], index[len(index) // 2 * 2 - i - 4], index[len(index) // 2 * 2 - i - 3], index[len(index) // 2 * 2 - i - 2], index[len(index) // 2 * 2 - i - 1], index[i-2], index[i-1],
            #        index[(i+1) % len(index)], index[(i+2) % len(index)], index[(len(index) // 2 * 2 - i + 1) % len(index)], index[(len(index) // 2 * 2 - i + 2) % len(index)], index[(len(index) // 2 * 2 - i + 3) % len(index)], index[(len(index) // 2 * 2 - i + 4) % len(index)], index[(len(index) // 2 * 2 - i + 5) % len(index)]]
            # seq = [index[i], index[i-1], index[(i+1) % len(index)], index[len(index) // 2 * 2 - i]]
            # seq = [index[i], index[i]-2, index[i]-1, (index[i]+2)%len(frames), (index[i]+1)%len(frames)]
        frame_seq = []
        for j in seq:
            frame_seq.append(frames[j])
        new_frames.append(frame_seq)
    return new_frames

def preprocessing(frames_dict, start_dict, mid_dict, end_dict):
    consistency_dict, new_ES_dict, new_ED_dict = CrossViewConsistencyFrameMapping(frames_dict, start_dict, mid_dict, end_dict)
    new_frame_dict = {}
    for k in consistency_dict.keys():
        new_frame = SymmetricSemanticReferenceSequenceSampling(frames_dict[k], consistency_dict[k], start_dict[k], new_ES_dict[k], new_ED_dict[k])
        new_frame_dict[k] = new_frame
    return new_frame_dict

def create_lists(frame_len, N, M, step, interval):
    result = []

    for i in range(frame_len):
        lst = []
        # 添加当前索引i
        lst.append(i)

        # 添加左边的N个索引
        for j in range(i-N, i):
            if j < 0:
                lst.append(j + frame_len)
            else:
                lst.append(j)

        # 添加右边的N个索引
        for j in range(i+1, i+N+1):
            if j >= frame_len:
                lst.append(j - frame_len)
            else:
                lst.append(j)

        # 添加左边的间隔索引
        # for j in range(i, i-interval*M-1, -step):
        #     idx = j - step
        #     if idx < 0:
        #         idx += frame_len
        #     lst.append(idx)
        #     if len(lst) == 2*N+M+1:
        #         break

        # 添加右边的间隔索引
        # for j in range(i, i+interval*M+1, step):
        #     idx = j + step
        #     if idx >= frame_len:
        #         idx -= frame_len
        #     lst.append(idx)
        #     if len(lst) == 1+2*N+2*M:
        #         break

        result.append(lst)

    return result


def create_lists_2(frame_len, N):
    results = []

    for i in range(frame_len):
        result = [i]

        near_area = sorted(list(set(range(i - N, i + N + 1))))
        while near_area[0] < 0:
            near_area = [i + 1 for i in near_area]

        while near_area[-1] >= frame_len:
            near_area = [i - 1 for i in near_area]

        if i in near_area:
            near_area.remove(i)

        result.extend(near_area)
        results.append(result)

    return results


def insert_and_return_value(lst, indexes, values):
    for index, value in zip(indexes, values):
        lst.insert(index, value)
    return lst

def stretch_videos(frames_dict, long_term=2, short_term=3, step=5, interval=5):
    new_frames_dict = {}
    for v, frames in frames_dict.items():
        frame_len = len(frames)
        # new_frames_idx = create_lists(frame_len, short_term, long_term, step, interval)
        new_frames_idx = create_lists_2(frame_len, short_term)
        new_frames = [[frames[i] for i in seq] for seq in new_frames_idx]
        new_frames_dict[v] = new_frames
    # for v, frames in frames_dict.items():
    #     idx = np.linspace(0, len(frames)-1, video_length).astype(np.uint8)
    #     new_frames_dict[v] = [frames[i] for i in idx]
    return new_frames_dict

def get_mean_and_std_metrics(patient_metrics):
    mean_and_std_metrics = {}
    for p, value in patient_metrics.items():
        dice = []
        nsd = []
        hd95 = []
        for v, metric in value.items():
            view_dice = []
            view_nsd = []
            view_hd95 = []
            for f, m in metric.items():
                view_dice.append(m[0])
                view_nsd.append(m[1])
                view_hd95.append(m[2])
            mean_view_dice = np.mean(view_dice)
            mean_view_nsd = np.mean(view_nsd)
            mean_view_hd95 = np.mean(view_hd95)
            dice.append(mean_view_dice)
            nsd.append(mean_view_nsd)
            hd95.append(mean_view_hd95)
        mean_dice = np.mean(dice)
        mean_nsd = np.mean(nsd)
        mean_hd95 = np.mean(hd95)
        mean_and_std_metrics[p] = (mean_dice, mean_nsd, mean_hd95)
    return mean_and_std_metrics

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


class Logger:
    def __init__(self, save_path, name):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename=os.path.join(save_path, f'{name}.log'), mode='w', encoding='UTF-8')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, text):
        self.logger.info(text)
        
    def delete(self):
        self.logger.handlers.clear()
    

if __name__ == '__main__':
    logger = Logger('test')
    logger.log('epoch 1')
