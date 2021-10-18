import os
import time
import errno
import logging
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing.pool import Pool
from time import sleep

def download_xmls(pid, gene_list, save_dir, base_url='https://www.proteinatlas.org/', overwrite=False):

    # dowload xmls
    for gene in tqdm(gene_list, total = len(gene_list), unit='files', postfix=pid):
        if "." in gene: gene = gene.split('.')[0]
        url = f'{base_url}{gene}.xml'
        try:
            out = requests.get(url)
            with open(f'{save_dir}{gene}.xml', 'w') as file:
                file.write(out.text)
        except: print(f'{url} broke...')

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

    parser.add_argument('source_file', type=str, help='Specify path to source_file.tsv')
    parser.add_argument('save_dir', type=str, help='Specify path output directory')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Specify number of workers (default: 1)')
    parser.add_argument('-o', '--overwrite', type=bool, default=False, help='Overwrite output dir and files if dir exists?')
    parser.add_argument('-url', '--base_url', type=str, default='https://www.proteinatlas.org/', help='Specify default URL')
    args = parser.parse_args()

    # create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=args.overwrite)
    print(f'Writing output xmls to {args.save_dir}...')

    print('Parent process %s.' % os.getpid())
    # load source xml to get ENSMBL IDs used in HPA
    df = pd.read_csv(args.source_file, sep='\t')
    genes = df['Gene'].unique()
    list_len = len(genes)

    p = Pool(args.workers)
    for i in range(args.workers):
        start = int(i * list_len / args.workers)
        end = int((i + 1) * list_len / args.workers)
        gene_list = genes[start:end]
        p.apply_async(
            download_xmls, args=(str(i), gene_list, args.save_dir, args.base_url, args.overwrite)
        )
        sleep(1)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(f"Total execution time: {time.time() - t00}")
    print(f"Time-per-file: {(time.time() - t00)/list_len}")
