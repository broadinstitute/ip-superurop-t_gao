"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import hpa_fewshot, voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex, run

from keras.models import load_model

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from neptune.new.types import File

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# # initialize Neptune.ai with API token
# with open('../../neptune-api-token.txt', 'r') as f:
#     run = neptune.init(
#         api_token=f.read(),
#         project='ip-superurop-tgao'
#     )

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    pretrained_path = _config['path']['init_path']
    model_cfg = _config['model']
    model = FewShotSeg(pretrained_path=pretrained_path, cfg=model_cfg)
    run['model/summary'] = model.get_summary()
    # run['model/pretrained_model'] = model.summary()
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    # run['model/summary'] = model.get_summary()
    model.train()

    # run['model/saved_model'] = model.module.summary()

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    _log.info('data_name is', data_name) # TODO: deleteme
    if data_name == 'VOC':
        make_data = voc_fewshot
        labels = CLASS_LABELS[data_name][_config['label_sets']]
    elif data_name == 'COCO':
        make_data = coco_fewshot
        labels = CLASS_LABELS[data_name][_config['label_sets']]
    elif data_name == 'HPA':
        make_data = hpa_fewshot
        rgb_dir = _config['path'][data_name]['rgb_dir']
        grayscale_dir = _config['path'][data_name]['grayscale_dir']
        # labels defined below
    else:
        raise ValueError('Wrong config for dataset!')

    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])

    base_dir = _config['path'][data_name]['data_dir']
    split = _config['path'][data_name]['data_split']
    to_tensor = ToTensorNormalize()
    max_iters = _config['n_steps'] * _config['batch_size']
    n_ways = _config['task']['n_ways']
    n_shots = _config['task']['n_shots']
    n_queries = _config['task']['n_queries']
    batch_size = _config['batch_size']
    shuffle = True
    num_workers = 1
    pin_memory = True
    drop_last = True

    if data_name == 'HPA':
        _log.info('data_name is', data_name) # TODO: deleteme
        dataset, labels = make_data(
            base_dir=base_dir, # _config['path'][data_name]['data_dir'],
            grayscale_dir=grayscale_dir,
            rgb_dir=rgb_dir,
            split=split, # _config['path'][data_name]['data_split'],
            transforms=transforms,
            to_tensor=to_tensor, # ToTensorNormalize(),
            max_iters=max_iters, # _config['n_steps'] * _config['batch_size'],
            n_ways=n_ways, # _config['task']['n_ways'],
            n_shots=n_shots, # _config['task']['n_shots'],
            n_queries=n_queries # _config['task']['n_queries']
        )
    else:
        _log.info('data_name is', data_name) # TODO: deleteme
        dataset = make_data(
            base_dir=base_dir, # _config['path'][data_name]['data_dir'],
            split=split, # _config['path'][data_name]['data_split'],
            transforms=transforms,
            to_tensor=to_tensor, # ToTensorNormalize(),
            labels=labels,
            max_iters=max_iters, # _config['n_steps'] * _config['batch_size'],
            n_ways=n_ways, # _config['task']['n_ways'],
            n_shots=n_shots, # _config['task']['n_shots'],
            n_queries=n_queries # _config['task']['n_queries']
        )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    for i_iter, sample_batched in enumerate(trainloader):

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    parameters = {
        'pretrained_path': pretrained_path,
        'model_cfg': model_cfg,
        'base_dir': base_dir,
        'split': split,
        'transforms': transforms,
        'to_tensor': to_tensor,
        'labels': labels,
        'max_iters': max_iters,
        'n_ways': n_ways,
        'n_shots': n_shots,
        'n_queries': n_queries,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        # 'optimizer': optimizer,
        # 'scheduler': scheduler,
        # 'criterion': criterion,
    }

    run['model/parameters'] = parameters

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                       query_images)
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss
        # run['train/loss'].log(query_loss)
        # run['train/align_loss'].log(align_loss)

        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            run['train/loss'].log(loss)
            run['train/align_loss'].log(align_loss)

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
            run['model/state_dict/snapshot/'].upload(os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    run['model/state_dict/final'].upload(os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
