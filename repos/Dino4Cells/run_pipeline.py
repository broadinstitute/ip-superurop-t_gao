import argparse
import yaml
import os

parser = argparse.ArgumentParser('Get embeddings from model')
parser.add_argument('--config', type=str, default='.', help='path to config file')
parser.add_argument('--gpus', type=str, default='.', help='Used GPUs, divided by commas (e.g., "1,2,4")')

args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))

num_gpus = len(args.gpus.split(','))

if config['pipeline']['run_DINO']:
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpus} python run_dino.py --config {args.config} --gpus {args.gpus}')

if config['pipeline']['run_get_embeddings']:
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpus} python run_get_embeddings.py --config {args.config}')

if config['pipeline']['run_plot_embeddings']:
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpus} python run_plot_embeddings.py --config {args.config}')

if config['pipeline']['run_classification']:
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpus} python run_classification.py --config {args.config}')

if config['pipeline']['run_visualize']:
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpus} python run_visualize.py --config {args.config}')
