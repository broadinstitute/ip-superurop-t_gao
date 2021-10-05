**Scripts to parse and download data from the Human Protein Atlas.**

The provided HPA_source_data.tsv contains high-level information on the HPA dataset and xml urls in particular. 

1. To download xmls run HPA_XML_download.py e.g.:
`python HPA_XML_download.py ../../datasets/HPA/pathology.tsv ../../datasets/HPA/XML/ -o TRUE -w 8`

2. To parse xmls and create a dictionary of genes (in Ensembl format) to a list of corresponding histology images for this gene run HPA_XML_parse. e.g.: `python HPA_XML_parse.py ../../datasets/HPA/XML/ ../../datasets/HPA/`

Alternatively, pass the fluoro_filter as input argument in order to filter for files containing red, blue, green or yellow in their filenames (aka fluorescent microscopy images).

3. To finally download images according to an input img_dictionary run HPA_IMG_download.py, e.g.:
`python HPA_img_download.py ../../datasets/HPA/HPA_histology_v1.pkl ../../datasets/HPA/data/ -o TRUE -w 8`
