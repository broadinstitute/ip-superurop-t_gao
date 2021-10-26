import os
import argparse
import yaml

parser = argparse.ArgumentParser('Visualize attention heads')
parser.add_argument('--config', type=str, default='.', help='path to config file')
parser.add_argument('--gpus', type=str, default='.', help='Used GPUs, divided by commas (e.g., "1,2,4")')

args = parser.parse_args()
config = yaml.safe_load(open(args.config, 'r'))

os.system(f"CUDA_VISIBLE_DEVICES={args.gpus} python visualize_attention.py --pretrained_weights {config['embedding']['pretrained_weights']} --image_path {config['visualize']['image_path']} --output_dir {config['plot_embedding']['output_name']} --arch {config['model']['arch']} --patch_size {config['model']['patch_size']} --image_size {config['embedding']['image_size']} --num_channels {config['model']['num_channels']} --loader {config['model']['datatype']}")
