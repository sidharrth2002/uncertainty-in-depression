# download the tar files from E-DAIC

# recursively download all files from https://dcapswoz.ict.usc.edu/wwwedaic/data/

import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    logging.info(f"Downloaded {url} to {local_filename}")
    return local_filename

def download_files(url, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    with requests.get(url) as r:
        r.raise_for_status()
        for line in r.text.splitlines():
            if line.startswith('<a href="') and line.endswith('.tar'):
                filename = line.split('"')[1].split('/')[-1]
                download_file(url + filename, os.path.join(local_dir, filename))

download_files('https://dcapswoz.ict.usc.edu/wwwedaic/data/', 'data')