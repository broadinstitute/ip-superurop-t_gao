# About

This subdirectory fetches, parses, and downloads biological cell image data from the [Human Protein Atlas](https://www.proteinatlas.org). This project uses the [pathology.tsv](https://www.proteinatlas.org/about/download) dataset.

Though this subdirectory may be used to obtain data from HPA, **an existing pre-generated dataset is available at a directory on the Broad Institute server**. The directory, `2021_10_06_HumanProteinAtlas_CiminiLab_TeresaGao`, may be accessed by connecting to the server `smb://hydrogen/imaging_analysis` via the [Broad Institute VPN](vpn.broadinstitute.org). In this directory are the JPG images as originally downloaded from HPA, grayscale masks, and color rotations (e.g., an image with blue nuclei, red microtubules, and green actin filaments recolored to have green nuclei, blue microtubules, and red actin filaments to counteract any possible unwanted effect of color on model training).

# Usage

*Source: https://github.com/WMAPernice/Dino4Cells*

The provided **HPA_source_data.tsv** contains high-level information on the HPA dataset and xml urls in particular.

First, download and unzip **pathology.tsv** from https://www.proteinatlas.org/about/download and save it in `datasets/HPA/`.

1. To download xmls, run HPA_XML_download.py e.g.:
`python HPA_XML_download.py datasets/HPA/pathology.tsv datasets/HPA/XML/ -o TRUE -w 8`

2. To parse xmls and create a dictionary of genes (in Ensembl format) to a list of corresponding histology images for this gene, run HPA_XML_parse. e.g.: `python HPA_XML_parse.py datasets/HPA/XML/ datasets/HPA/`.

3. Finally, to download images according to an input img_dictionary run HPA_IMG_download.py, e.g.:
`python HPA_img_download.py datasets/HPA/HPA_histology_v1.pkl datasets/HPA/data/ -o TRUE -w 8`
