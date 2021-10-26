import ast
import urllib.request
import pandas as pd
from pathlib import Path

def grab_images(df_path, save_dir, verbose=True):
    """
    Given path to output of running hpa_xml_parse.py on result of hpa_xml_download.py,
    download and save JPGs with a single protein location in directory save_dir;
    grouped by protein location.
    """
    img_df = pd.read_csv(df_path)
    Path(save_dir).mkdir(exist_ok=True)

    for index, row in img_df.iterrows():
        if verbose:
            print(f'processing row index {index}...')

        protein_locations = ast.literal_eval(row['protein_location'])
        if len(protein_locations) == 1:
            gene_name = row['gene'].strip()
            protein_subdirectory = save_dir + '/' + protein_locations[0].strip().replace(' ', '-')
            cell_type = row['cell_type'].strip().replace(' ', '-')
            image_url = row['img_file'].strip()
            save_file = protein_subdirectory + '/' + gene_name + '_' + cell_type + '_' + image_url.split("/")[-1]

            Path(protein_subdirectory).mkdir(exist_ok=True)
            urllib.request.urlretrieve(image_url, save_file)

            if verbose:
                print(f'\timage at {image_url} saved to {save_file}')

    print('done :)')

if __name__ ==  '__main__':
    df_path = 'datasets/HPA/HPA_img_df.csv' # result of running hpa_xml_parse.py on result of hpa_xml_download.py
    save_dir = 'datasets/HPA/JPG'
    grab_images(df_path, save_dir)