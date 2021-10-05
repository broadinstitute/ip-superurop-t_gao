import os
import time
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from multiprocessing.pool import Pool
from HPA_utils import pkl_load

tqdm.pandas()


def download_images_from_df(pid, gene, ids, save_dir = '../datasets/HPA/data/', overwrite=True, verbose=True):

    out = save_dir+gene
    if verbose: print('creating dir:', out)
    Path(out).mkdir(parents=True, exist_ok=overwrite)

    if verbose: print(f"downloading {len(ids)} images for {gene}")
    # TODO: work on creating nicely nested tqdm progress panels...
    # for url in tqdm(ids, total = len(ids), unit="files", postfix=pid):
    for url in ids:
        try:
            for channel in ['red','green','yellow']:
                url = url.replace('_blue_red_green', f'_{channel}')
                r = requests.get(url)
                im = Image.open(BytesIO(r.content))
                im.save(os.path.join(out, url.split('/')[-1]), 'JPEG')
        except Exception as e:
            print(f'{e}')
            print(f'{url} broke...')

def download_images(pid, gene, ids, save_dir = '../datasets/HPA/data/', overwrite=True, verbose=True):

    out = save_dir+gene
    if verbose: print('creating dir:', out)
    Path(out).mkdir(parents=True, exist_ok=overwrite)

    if verbose: print(f"downloading {len(ids)} images for {gene}")
    # TODO: work on creating nicely nested tqdm progress panels...
    # for url in tqdm(ids, total = len(ids), unit="files", postfix=pid):
    for url in ids:
        try:
            for channel in ['red','green','yellow']:
                url = url.replace('_blue_red_green', f'_{channel}')
                r = requests.get(url)
                im = Image.open(BytesIO(r.content))
                im.save(os.path.join(out, url.split('/')[-1]), 'JPEG')
        except Exception as e:
            print(f'{e}')
            print(f'{url} broke...')

def download_single_image_from_df(img_file, save_dir):
    try:
        image = []
        for channel in ['red','green','blue','yellow']:
            new_img_file = img_file.replace('blue',channel)
            r = requests.get(new_img_file)
            im = Image.open(BytesIO(r.content))
            # im = im.resize((512,512)).convert('L')
            im = im.convert('L')
            image.append(im)
        image = Image.merge('RGBA', image)
        image.save(os.path.join(save_dir, '_'.join(img_file.split('/')[-2:])), 'png')
        print(f'downloading {new_img_file.replace("_yellow","")}...')
    except Exception as e:
        print(f'{e}')
        print(f'{img_file} broke...')

def download_from_df(pid, img_dict, save_dir, overwrite=False):
    img_dict.img_file.progress_apply(download_single_image_from_df, args=[save_dir])

def download_from_dict(pid, img_dict, save_dir, keys=None, overwrite=False):
    if keys is not None: img_dict = {k: img_dict[k] for k in keys}
    for gene, ids in tqdm(img_dict.items(), total=len(list(img_dict.keys())), unit='files', postfix=pid):
        try: download_images(pid, gene, ids[:10], save_dir = save_dir, overwrite=overwrite)
        except FileExistsError: print(f'{save_dir+gene} already exists. Skipping {gene}...')

if __name__ == '__main__':
    t00 = time.time()
    # parameters:
    parser = argparse.ArgumentParser()

    class handledict(argparse.Action):
        def __call__(self, parser, namespace, instring, option_string=None):
            my_dict = {}
            for keyval in instring.split(","):
                print(keyval)
                key,val = keyval.split(":")
                my_dict[key] = val
            setattr(namespace, self.dest, my_dict)

    parser.add_argument('img_dict', type=str, help='Specify path to image dictionary pickle file, e.g. img_dict.pkl')
    parser.add_argument('save_dir', type=str, help='Specify path output directory')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Specify number of workers (default: 1)')
    parser.add_argument('-o', '--overwrite', type=bool, default=False, help='Overwrite output dir and files if dir exists?')
    args = parser.parse_args()

    # create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=args.overwrite)
    print(f'Downloading images to {args.save_dir}...')

    print('Parent process %s.' % os.getpid())
    # load source xml to get ENSMBL IDs used in HPA

    # Changed to csv to fit the new XML parse
    d = pd.read_csv(args.img_dict)
    # d = pkl_load(args.img_dict)
    # genes = list(d.keys())[:4] # Is this 4 intentional?
    # genes = list(d.keys())
    # list_len = len(genes)
    list_len = len(d)

    p = Pool(args.workers)
    jump = int(list_len / args.workers)
    jumps = [[0, jump]]
    for w in range(1, args.workers):
        jumps.append((jumps[-1][-1], jumps[-1][1] + jump))
    print(jumps)
    for i in range(args.workers):
        start = jumps[i][0]
        end = jumps[i][1]
        subset_to_download = d.iloc[start:end]

        p.apply_async(
            # download_from_dict, args=(str(i), d, args.save_dir, gene_list, args.overwrite)
            download_from_df, args=(str(i), subset_to_download, args.save_dir, args.overwrite)
        )
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"Total execution time: {time.time() - t00}")
    print(f"Time-per-file: {(time.time() - t00)/list_len}")
