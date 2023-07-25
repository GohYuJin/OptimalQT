import os
import argparse
import numpy as np
from PIL import Image
from DiffJPEG import quality_to_factor
from helpers import create_default_qtables

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('path', type=str,
                    help='path to imagenet folder')

args = parser.parse_args()
root = args.path

HQ_files = open("imagenetHQ.csv", "w")
LQ_files = open("imagenetLQ.csv", "w")

y_table, c_table = create_default_qtables()
y_max = y_table.numpy().max() * quality_to_factor(90)
c_max = c_table.numpy().max() * quality_to_factor(90)

for split_folder in ["train", "val"]:
    for idx, class_folder in enumerate(os.listdir(os.path.join(root, split_folder))):
        for file in os.listdir(os.path.join(root, split_folder, class_folder)):
            rel_path = os.path.join(split_folder, class_folder, file)
            im = Image.open(os.path.join(root, rel_path))
            
            try:
                im.quantization
            except:
                continue
                
            if 0 in im.quantization and 1 in im.quantization:
                if (np.array(im.quantization[0]) <= y_max).all() and (np.array(im.quantization[1]) <= c_max).all():
                    HQ_files.write(rel_path + "\n")
                else:
                    LQ_files.write(rel_path + "\n")
        print("Splitting images for", split_folder, ":", idx, "/1000")
        
HQ_files.close()
LQ_files.close()

hq = pd.read_csv("imagenetHQ.csv", header=None)
hq.columns = ["files"]
hq_val = hq[hq["files"].str.startswith("val")]
hq_val.to_csv("imagenetHQ_val.csv", index=False)
hq_train = hq[~hq["files"].isin(hq_val["files"])]
hq_train.to_csv("imagenetHQ_train.csv", index=False)