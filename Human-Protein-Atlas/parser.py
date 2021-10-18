import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

xml_file = "nuclear-bodies-mitochondria.xml"
use_genes_subset = False
genes_subset = {"ZNF26"}

tree = ET.parse(xml_file)
root = tree.getroot()

for entry in root.findall("entry"):
    gene_name = entry.find("name").text
    if not use_genes_subset or gene_name in genes_subset:
        for imageUrl in entry.findall("./antibody/cellExpression/subAssay/data/assayImage/image/imageUrl"):
            Path(gene_name).mkdir(exist_ok=True)
            urllib.request.urlretrieve(imageUrl.text, gene_name + "/" + imageUrl.text.split("/")[-1])