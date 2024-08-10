from M2EchoSeg import M2EchoSeg
from trainer import Trainer
from predictor import Predictor
from monai.utils import set_determinism
from utils import TrainConfig
import wandb

import argparse


if __name__ == '__main__':
    set_determinism(seed=0)
    parser = argparse.ArgumentParser(description='M2Seg')
    parser.add_argument('--method_index', type=int, default=0, help='Method index')
    parser.add_argument('--dataset', type=int, default=0, help='Dataset index')
    parser.add_argument('--is_cv', type=bool, default=True, help='Cross validation')
    parser.add_argument('--fold', type=int, default=5, help='K-fold')
    parser.add_argument('--N', type=int, default=32)
    parser.add_argument('--divloss', type=bool, default=True)
    opt = parser.parse_args()
    method_index = opt.method_index
    method_names = ['M2EchoSeg']
    datasets = ['HCM', 'CAMUS', 'HCM-ext']
    views = {'HCM': [0, 1, 2], 'CAMUS': [0, 1], 'HCM-ext': [0, 1, 2]}
    # wandb.login()
        
    config = TrainConfig(project_name=f'{method_names[method_index]}',
                         is_cv=opt.is_cv,
                         cv_folds=opt.fold,
                         dataset=datasets[opt.dataset],
                         views=views[datasets[opt.dataset]],
                         is_video=True,
                         image_size=(192, 192),
                         mv=True,
                         multi_view=True,
                         ssim_loss=True,
                         batch_size=16,
                         lr_scheduler='MultiStepLR',
                         optimizer='Adam',
                         learning_rate=1e-3,
                         video_length=7,
                         activation='relu',
                         ext_test=True if opt.dataset == 3 else False,
                         divloss=opt.divloss,
                         frame_align='ES'
                         )
    model = M2EchoSeg(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        act=config.activation,
        dropout_prob=config.dropout,
        features=config.features,
        dropout_dim=2,
        bias=True,
        n_views=len(views[datasets[opt.dataset]]),
        video_length=config.video_length,
        N=opt.N
    ) # 10.1531M

    # trainer = Trainer(
    #     method_name=f'{method_names[method_index]}', model=model, config=config, debug=True)
    # trainer.train()
    pred = Predictor(method_name=f'{method_names[method_index]}', model=model, config=config, debug=True)
    pred.prediction()
