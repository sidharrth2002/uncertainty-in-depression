import os
from tqdm import tqdm
import argparse
import tarfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root-dir", type=str, default="./data/E-DAIC/backup/")
    parser.add_argument("--dest-dir", type=str, default="./data/E-DAIC/data/")
    args = parser.parse_args()

    tar_files = sorted( os.listdir(args.root_dir) )
    # only keep stuff that ends with .tar.gz
    tar_files = [tar_file for tar_file in tar_files if tar_file.endswith('.tar.gz')]

    for tar_file in tqdm(tar_files):
       print(tar_file)
       tar_path = os.path.join(args.root_dir, tar_file)
       print(tar_path)
       
       
       
       with tarfile.open(tar_path, 'r') as tar:
           tar.extractall(args.dest_dir)
                