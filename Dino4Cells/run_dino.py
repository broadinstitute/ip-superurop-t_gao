import argparse
import yaml
import os

parser = argparse.ArgumentParser('Get embeddings from model')
parser.add_argument('--config', type=str, default='.', help='path to config file')
parser.add_argument('--gpus', type=str, default='.', help='Used GPUs, divided by commas (e.g., "1,2,4")')

args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))
if 'local_crops_scale' in config['model']:
    local_crops_scale = config['model']['local_crops_scale']
else:
    local_crops_scale = "0.05 0.4"
if 'epochs' in config['model']:
    epochs = config['model']['epochs']
else:
    epochs = 100

num_gpus = len(args.gpus.split(','))
command = f'CUDA_VISIBLE_DEVICES={args.gpus} python -m torch.distributed.launch --nproc_per_node={num_gpus} main_dino.py --arch {config["model"]["arch"]} --output_dir {config["model"]["output_dir"]} --data_path {config["model"]["data_path"]} --datatype {config["model"]["datatype"]} --saveckp_freq {config["model"]["saveckp_freq"]} --batch_size_per_gpu {config["model"]["batch_size_per_gpu"]} --num_channels {config["model"]["num_channels"]} --patch_size {config["model"]["patch_size"]} --local_crops_scale {local_crops_scale} --epochs {epochs}'
os.system(command)
