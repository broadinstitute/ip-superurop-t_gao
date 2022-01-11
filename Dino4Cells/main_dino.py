# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import cell_utils #(!)
from file_dataset import ImageFileList, RGBA_loader, protein_loader, pandas_reader_website, both_loader, protein_transparency_loader, RGBA_norm_loader, threshold_loader, threshold_new_channel_loader,single_cell_mirror_no_gap_loader
from file_dataset import red_channel_loader, green_channel_loader, blue_channel_loader, yellow_channel_loader, no_reticulum_loader, single_cell_mirror_loader, single_cell_resize_loader, single_cell_resize_keep_aspect_loader, single_cell_resize_no_mask_loader, single_cell_resize_with_noise_loader, tile_loader
from functools import partial #(!)
from archs import xresnet as cell_models #(!)
from cell_dataset import dino_dataset
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

cell_archs = sorted(name for name in cell_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(cell_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] + torchvision_archs + cell_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--num_channels', default=3, type=int,
        help="""Number of channels for the Vision transformer""")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--loader', default="folder", type=str, choices = ['folder', 'png_file_list', 'tif_file_list'], help='Type of data loader')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--datatype', default='RGB', type=str,
        choices=['RGB', 'CellNet', 'BBBC021', 'HPA','HPA_single_cell_mirror_no_gap','HPA_threshold_preprocessing_new_channel','HPA_single_cell_mirror','HPA_single_cell_resize_jigsaw','HPA_single_cell_resize_rotate_dihedral','HPA_single_cell_resize_rotate','HPA_single_cell_resize_warp','HPA_single_cell_resize_partial_warp','HPA_single_cell_random_resize_partial_warp','HPA_single_cell_resize_partial','HPA_single_cell_resize_partial_random','HPA_tile','HPA_single_cell_resize','HPA_single_cell_resize_noise','HPA_single_cell_resize_no_mask','HPA_single_cell_resize_keep_aspect','HPA_threshold_preprocessing','HPA_threshold_protein','HPA_threshold_protein_low','HPA_single_cell_resize_combo_v1','HPA_combo_v2', 'HPA_combo_v1','HPA_combo_protein_v1', 'HPA_no_reticulum','HPA_modify_jitter','HPA_remove_blur_grey_solar_aug', 'HPA_rescaling_protein_aug', 'HPA_min_max_norm', 'HPA_remove_flip_aug','HPA_remove_color_jitter_aug','HPA_remove_solarization_aug','HPA_remove_blur_aug','HPA_remove_channel', 'HPA_remove_grey_aug', 'BBBC021_4_channels','protein','red_channel','blue_channel','green_channel','yellow_channel', 'HPA_both', 'HPA_rotation','HPA_alpha_mask_protein'], help="""Name of the source dataset; will be used to set dataloader and tfms.""")
    parser.add_argument('--root_dir_path', default="/home/ubuntu/data/CellNet_data/Hirano3D_v2.0/data/", type=str,
                        help='root_dir_path for cells_dataloader.')
    parser.add_argument('--RGBmode', default=False, type=utils.bool_flag, help="""enforce 3-channels""")
    parser.add_argument('--scale-factor', default=1.0, type=float, help="""Set factor by which to resize source images.""")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
        args.datatype,
        args.scale_factor
    )
    if args.datatype == 'CellNet':
        dataset = dino_dataset(args.data_path, args.root_dir_path, transform=transform,
                  RGBmode = args.RGBmode)
    elif args.datatype == 'BBBC021':
        dataset = ImageFileList(args.data_path, transform=transform)
    elif args.datatype == 'BBBC021_4_channels':
        dataset = ImageFileList(args.data_path, transform=transform)
    elif args.datatype in ['HPA','HPA_threshold_protein_low','HPA_single_cell_resize_partial','HPA_single_cell_resize_partial_warp','HPA_single_cell_random_resize_partial_warp','HPA_single_cell_resize_partial_random','HPA_threshold_protein','HPA_combo_v1','HPA_combo_protein_v1', 'HPA_rotation','HPA_remove_blur_grey_solar_aug','HPA_modify_jitter','HPA_rescaling_protein_aug', 'HPA_remove_grey_aug','HPA_remove_channel','HPA_remove_flip_aug','HPA_remove_color_jitter_aug','HPA_remove_solarization_aug','HPA_remove_blur_aug']:
        dataset = ImageFileList(args.data_path, transform=transform, loader = RGBA_loader, flist_reader = pandas_reader_website)
    elif args.datatype in ['HPA_min_max_norm','HAP_combo_v2']:
        dataset = ImageFileList(args.data_path, transform=transform, loader = RGBA_norm_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_alpha_mask_protein':
        dataset = ImageFileList(args.data_path, transform=transform, loader = protein_transparency_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_both':
        dataset = ImageFileList(args.data_path, transform=transform, loader = both_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'protein':
        dataset = ImageFileList(args.data_path, transform=transform, loader = protein_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'red_channel':
        dataset = ImageFileList(args.data_path, transform=transform, loader = red_channel_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'green_channel':
        dataset = ImageFileList(args.data_path, transform=transform, loader = green_channel_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'blue_channel':
        dataset = ImageFileList(args.data_path, transform=transform, loader = blue_channel_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'yellow_channel':
        dataset = ImageFileList(args.data_path, transform=transform, loader = yellow_channel_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_no_reticulum':
        dataset = ImageFileList(args.data_path, transform=transform, loader = no_reticulum_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_threshold_preprocessing':
        dataset = ImageFileList(args.data_path, transform=transform, loader = threshold_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_threshold_preprocessing_new_channel':
        dataset = ImageFileList(args.data_path, transform=transform, loader = threshold_new_channel_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_single_cell_mirror_no_gap':
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_mirror_no_gap_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_single_cell_mirror':
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_mirror_loader, flist_reader = pandas_reader_website)
    elif args.datatype in ['HPA_single_cell_resize','HPA_single_cell_resize_rotate','HPA_single_cell_resize_warp','HPA_single_cell_resize_rotate_dihedral','HPA_single_cell_resize_jigsaw','HPA_single_cell_resize_combo_v1']:
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_resize_loader, flist_reader = pandas_reader_website)
    elif args.datatype in ['HPA_tile']:
        dataset = ImageFileList(args.data_path, transform=transform, loader = tile_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_single_cell_resize_keep_aspect':
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_resize_keep_aspect_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_single_cell_resize_no_mask':
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_resize_no_mask_loader, flist_reader = pandas_reader_website)
    elif args.datatype == 'HPA_single_cell_resize_noise':
        dataset = ImageFileList(args.data_path, transform=transform, loader = single_cell_resize_with_noise_loader, flist_reader = pandas_reader_website)

    else:
        print('Datatype not supported!')
        quit()
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
            in_chans=args.num_channels,
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size, in_chans=args.num_channels)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    elif args.arch in cell_models.__dict__.keys():
        student = partial(cell_models.__dict__[args.arch], c_in=args.num_channels)(False)
        teacher = partial(cell_models.__dict__[args.arch], c_in=args.num_channels)(False)
        embed_dim = student[-1].weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)




class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, datatype, scale_factor=1):
        self.datatype = datatype
        if datatype == "RGB":
            print('RGB mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

        elif datatype in ["BBBC021"]:
            print('BBBC021 mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'BBBC021_4_channels':
            print('BBBC021 4 channels mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(np.concatenate((np.array(x)[:,:,[0,1]], np.zeros((x.size[0], x.size[0], 1)), np.array(x)[:,:,[2]]), axis=2).astype(np.uint8))),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(
                    np.concatenate((np.array(x)[:,:,[0,1]], np.zeros((x.size[0], x.size[0], 1)),
                                    np.array(x)[:,:,[2]]), axis=2).astype(np.uint8))),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.Lambda(lambda x: Image.fromarray(np.concatenate((np.array(x)[:,:,[0,1]], np.zeros((x.size[0], x.size[0], 1)), np.array(x)[:,:,[2]]), axis=2).astype(np.uint8))),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

        elif datatype in ['protein','red_channel', 'blue_channel', 'green_channel', 'yellow_channel']:
            print('protein mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                cell_utils.Get_specific_channel(1),
                transforms.ToTensor(),
                transforms.Normalize((0.485), (0.229)),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_both','HPA_no_reticulum']:
            print('HPA both mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_combo_v2']:
            print('HPA combo v2')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                cell_utils.Change_brightness(p=0.8),
                cell_utils.Change_contrast(p=0.8),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
        elif datatype in ['HPA_combo_v1','HPA_single_cell_resize_combo_v1']:
            print('HPA combo v1')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                cell_utils.Change_brightness(p=0.8),
                cell_utils.Change_contrast(p=0.8),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
        elif datatype == 'HPA_combo_protein_v1':
            print('HPA combo protein v1')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                cell_utils.Change_brightness(p=0.8),
                cell_utils.Change_contrast(p=0.8),
            ])
            normalize = transforms.Compose([
                cell_utils.Get_specific_channel(1),
                transforms.ToTensor(),
                transforms.Normalize((0.485), (0.229)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
        elif datatype == 'HPA_remove_channel':
            print('HPA remove channel')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

        elif datatype == 'HPA_modify_jitter':
            print('HPA modify jitter')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                cell_utils.change_brightness(p=0.8),
                cell_utils.change_contrast(p=0.8),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_rescaling_protein_aug':
            print('HPA rescaling protein aug mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])

        elif datatype == 'HPA_remove_blur_grey_solar_aug':
            print('HPA remove blur, greyscale and solarization mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.remove_channel(0.2),
                cell_utils.rescale_protein(p=0.2),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
        elif datatype == 'HPA_rotation':
            print('HPA rotation mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomRotation(degrees=90, expand=False),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_alpha_mask_protein':
            print('HPA alpha mask protein')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_remove_blur_aug':
            print('HPA remove blur mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                normalize,
            ])
        elif datatype == 'HPA_remove_solarization_aug':
            print('HPA remove solarization mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_remove_color_jitter_aug':
            print('HPA remove color jitter mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_remove_flip_aug':
            print('HPA remove flip mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_remove_grey_aug':
            print('HPA without greyscale augmentation mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_min_max_norm':
            print('HPA min max norm mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype == 'HPA_threshold_protein':
            print('HPA threshold protein mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.Threshold_protein(p=0.8),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.Threshold_protein(p=0.8),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.Threshold_protein(p=0.8),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_threshold_preprocessing']:
            print('HPA threshold channel mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_threshold_preprocessing_new_channel']:
            print('HPA new preprocessing mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_5_channels(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_5_channels(p=0.2),
            ])
            normalize = transforms.Compose([
                cell_utils.Permute(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482, 0.1482)),
            ])
            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop_for_5_channels(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                cell_utils.GaussianBlur_for_5_channels(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop_for_5_channels(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                cell_utils.GaussianBlur_for_5_channels(0.1),
                cell_utils.Solarization_for_5_channels(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop_for_5_channels(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                cell_utils.GaussianBlur_for_5_channels(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_jigsaw']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Jigsaw(p=0.8),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Jigsaw(p=0.8),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_rotate_dihedral']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.Rotate_dihedral(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.Rotate_dihedral(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.Rotate_dihedral(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_partial_random']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_random_resize_partial_warp']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_random_resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_partial_warp']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize','HPA_single_cell_resize_partial']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                # cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_warp']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Warp_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_tile']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_resize_rotate']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.Rotate_single_cell(),
                cell_utils.Single_cell_Resize(),
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA_single_cell_mirror','HPA_single_cell_mirror_no_gap','HPA_single_cell_resize_keep_aspect','HPA_single_cell_resize_no_mask','HPA_single_cell_resize_noise']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        elif datatype in ['HPA']:
            print('HPA mode')
            flip_and_color_jitter = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                cell_utils.RandomGrayscale_for_RGBA(p=0.2),
            ])
            normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2145, 0.2145, 0.2145, 0.2145), (0.1483, 0.1482, 0.1482, 0.1482)),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                cell_utils.Solarization_for_RGBA(0.2),
                normalize,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ])
        else:
            print('CellNet mode')
            dp=1e6
            sd=0.7

#             color_jitter = transforms.Compose([
#                 transforms.RandomApply(
#                     [cell_utils.ColorJitter_for_RGBA(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                     p=0.8
#                 ),
#                 cell_utils.RandomGrayscale_for_RGBA(p=0.2),
#             ])


            augment = transforms.Compose([
                cell_utils.rnd_dihedral(),
#                 cell_utils.random_rotate(),
            ])
            normalize = transforms.Compose([
#                 transforms.ToTensor(),
                cell_utils.self_normalize(),
            ])

            # first global crop
            self.global_transfo1 = transforms.Compose([
                cell_utils.FA_resize(np.int(512*scale_factor)),
                normalize,
#                 transforms.ToPILImage(),
#                 cell_utils.RandomResizedCrop(np.int(224*scale_factor), scale=global_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.RandomResizedCenterCrop(np.int(224*scale_factor), scale=global_crops_scale, depth=dp, s=sd),
#                 color_jitter,
#                 utils.GaussianBlur(1.0), #1.0
#                 transforms.ToTensor(),
                augment,
            ])
            # second global crop
            self.global_transfo2 = transforms.Compose([
                cell_utils.FA_resize(np.int(512*scale_factor)),
                normalize,
#                 transforms.ToPILImage(),
#                 cell_utils.RandomResizedCrop(np.int(224*scale_factor), scale=global_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.RandomResizedCenterCrop(np.int(224*scale_factor), scale=global_crops_scale, depth=dp, s=sd),
#                 color_jitter,
#                 utils.GaussianBlur(0.1),
#                 cell_utils.Solarization_for_RGBA(0.2),
#                 transforms.ToTensor(),
                augment,
            ])
            # transformation for the local small crops
            self.local_crops_number = local_crops_number
            self.local_transfo = transforms.Compose([
                cell_utils.FA_resize(np.int(512*scale_factor)),
                normalize,
#                 transforms.ToPILImage(),
#                 cell_utils.RandomResizedCrop(np.int(96*scale_factor), scale=local_crops_scale, interpolation=Image.BICUBIC),
                cell_utils.RandomResizedCenterCrop(np.int(96*scale_factor), scale=local_crops_scale, depth=dp, s=sd),
#                 color_jitter,
#                 utils.GaussianBlur(p=0.5), #0.5
#                 transforms.ToTensor(),
                augment,
            ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
