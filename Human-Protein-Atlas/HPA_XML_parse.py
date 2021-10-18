import os
import argparse
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
from HPA_utils import pkl_save, pkl_load, load_xml
import numpy as np


# string-based filters for image urls:
def histo_filter(fn):
    b = any(x in fn for x in ['blue', 'green', 'red', 'yellow']) or 'protein_array' in fn
    return not b

def fluoro_filter(fn):
    b = any(x in fn for x in ['blue', 'green', 'red', 'yellow'])
    return b

def bgr_filter(fn):
    b = b = any((x in fn for x in ['blue', 'green', 'red']) and (x not in fn for x in ['yellow', 'protein_array']))
    return b

def preprocess(img):
    img = img.replace('_blue_red_green','_blue')
    return img

def get_ims_list(xml_dir, save_dir, fltr = bgr_filter, preprocess = None):
    """
    Parse xmls and apply filters
    """
    d = {}
    bad_xmls = []
    img_list = []
    all_locations = []
    cell_types = []
    genes = []
    xmls = os.listdir(xml_dir)
    no_loc_images = []
    no_loc_xmls = []
    for xml in tqdm(xmls, total=len(xmls), unit='files'):
        try:
            tree, root = load_xml(xml_dir+xml)
            gene = root.find('./entry/identifier').attrib['id']
            cell_expressions = []
            for antibody in list(root.iter('antibody')):
                cell_expressions.extend(list(antibody.iter('cellExpression')))
            cell_expressions = [c for c in cell_expressions if c.attrib['technology'] == 'ICC/IF']
            subAssays = []
            for c in cell_expressions:
                subAssays.extend([s for s in list(c.iter('subAssay')) if s.attrib['type'] == 'human'])
            data_elements = []
            for s in subAssays:
                data_elements.extend(list(s.iter('data')))
            temp_locations = [d.find('location').text if type(d.find('location')) != type(None) else d.find('location') for d in data_elements]

            for d in data_elements:
                for img_file in d.iter('image'):
                    try:
                        im = img_file.find('imageUrl').text
                        if fltr(im):
                            if preprocess: im = preprocess(im)
                            locations = [l.text for l in d.iter('location')]
                            cell_type = d.find('cellLine').text
                            all_locations.append(locations)
                            cell_types.append(cell_type)
                            img_list.append(im)
                            genes.append(gene)
                    except:
                        no_loc_images.append(im)
                        no_loc_xmls.append(xml)
        except Exception as e:
            print(e)
            bad_xmls.append(xml)
    d = pd.DataFrame(zip(img_list,
                         all_locations,
                         cell_types,
                         genes),
                     columns = [
                         'img_file',
                         'protein_location',
                         'cell_type',
                         'gene',
                         ])

    print(f'{len(xmls) - len(c)} of {len(xmls)} parsed successfully.')
    pkl_save(d, save_dir+"HPA_img_dict.pkl")
    d.to_csv(f'{save_dir}/HPA_img_df.csv', index=False)

    pd.DataFrame(bad_xmls, columns=['Gene']).to_csv(save_dir+'HPA_failed_xmls.tsv', sep='\t', index=False)
    pd.DataFrame(zip(no_loc_images, no_loc_xmls), columns=['img_file','xml']).to_csv(save_dir+'HPA_no_location_xmls.tsv', sep='\t', index=False)

if __name__ == '__main__':
    # TODO: implement multiprocessing...

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

    parser.add_argument('xml_dir', type=str, help='Specify path to xml_dir')
    parser.add_argument('save_dir', type=str, help='Specify path output directory')
    parser.add_argument('-f', '--filter', type=str, default='histo_filter', help='Specify filter function. Options: histo_filter (default), fluore_filter')
    parser.add_argument('-b', '--blue', action='store_true', help='Preprocess file name to only return blue channel')

    args = parser.parse_args()

    # fltrs = {'histo_filter': histo_filter,
    #          'fluoro_filter': fluoro_filter}
    # fltr = fltrs[args.filter]
    fltr = bgr_filter
    get_ims_list(args.xml_dir, args.save_dir, fltr=fltr, preprocess= preprocess if args.blue else None)
