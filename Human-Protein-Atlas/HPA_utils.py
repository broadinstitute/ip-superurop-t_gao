import pickle
import xml.etree.ElementTree as ET

def load_xml(path):
    parser = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(path, parser=parser)
    root = tree.getroot()
    return tree, root
    
def pkl_save(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def pkl_load(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle) 
    return data