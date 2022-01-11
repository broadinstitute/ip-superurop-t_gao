import numpy as np

num_to_protein_full = {}
num_to_protein_full[0] = 'nucleoplasm'
num_to_protein_full[1] = 'nuclear membrane'
num_to_protein_full[2] = 'nucleoli'
num_to_protein_full[3] = 'nucleoli fibrillar center'
num_to_protein_full[4] = 'nuclear speckles'
num_to_protein_full[5] = 'nuclear bodies'
num_to_protein_full[6] = 'endoplasmic reticulum'
num_to_protein_full[7] = 'golgi apparatus'
num_to_protein_full[8] = 'peroxisomes'
num_to_protein_full[9] = 'endosomes'
num_to_protein_full[10] = 'lysosomes'
num_to_protein_full[11] = 'intermediate filaments'
num_to_protein_full[12] = 'actin filaments'
num_to_protein_full[13] = 'focal adhesion sites'
num_to_protein_full[14] = 'microtubules'
num_to_protein_full[15] = 'microtubule ends'
num_to_protein_full[16] = 'cytokinetic bridge'
num_to_protein_full[17] = 'mitotic spindle'
num_to_protein_full[18] = 'microtubule organizing center'
num_to_protein_full[19] = 'centrosome'
num_to_protein_full[20] = 'lipid droplets'
num_to_protein_full[21] = 'plasma membrane'
num_to_protein_full[22] = 'cell junctions'
num_to_protein_full[23] = 'mitochondria'
num_to_protein_full[24] = 'aggresome'
num_to_protein_full[25] = 'cytosol'
num_to_protein_full[26] = 'cytoplasmic bodies'
num_to_protein_full[27] = 'rods & rings'

protein_to_num_full = {v.lower(): k for (k, v) in num_to_protein_full.items()}

num_to_protein_5k = {}
num_to_protein_5k[0] = 'nucleoplasm'
num_to_protein_5k[1] = 'plasma membrane'
num_to_protein_5k[2] = 'mitochondria'
num_to_protein_5k[3] = 'cytosol'
num_to_protein_5k[4] = 'vesicles'

protein_to_num_5k = {v.lower(): k for (k, v) in num_to_protein_5k.items()}

num_to_other_protein_5k = {}
num_to_other_protein_5k[0] = 'golgi apparatus'
num_to_other_protein_5k[1] = 'nuclear speckles'
num_to_other_protein_5k[2] = 'nuclear bodies'
num_to_other_protein_5k[3] = 'nucleoli'
num_to_other_protein_5k[4] = 'endoplasmic reticulum'

other_protein_to_num_5k = {v.lower(): k for (k, v) in num_to_other_protein_5k.items()}
