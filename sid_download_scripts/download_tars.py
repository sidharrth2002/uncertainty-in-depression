import bs4
import requests
import multiprocessing
import os
import sys
sys.setrecursionlimit(10000)

url = 'https://dcapswoz.ict.usc.edu/wwwedaic/data/'

download_path = "/home/sn666/data/edaic"

def download_link_to_file(link, file_path):
    # check if link is in log.txt
    with open(os.path.join(download_path, 'log.txt'), 'r') as f:
        if link in f.read():
            return
    print(f'Started downloading {link}')
    with open(file_path, 'wb') as f:
        f.write(requests.get(link).content)
        # once the file is downloaded, write to log so we don't download it again
        print(f'Done downloading {link}')
        with open(os.path.join(download_path, 'log.txt'), 'a') as f:
            f.write(link + '\n')
    # wget.download(link, out=file_path, no_check_certificate=True)
    return True


def download(l):
    # print(f"Downloading {l.get_text()}")
    return download_link_to_file(url + l.get('href'), download_path + l.get_text())


if __name__ == '__main__':
    # download all files in the directory
    r = requests.get(url)
    data = bs4.BeautifulSoup(r.text, 'html.parser')
    function_arguments = []
    for l in data.find_all('a'):
        if l.get('href') != '/' and l.parent.name == 'td' and l.get_text().split('_')[0].isdigit():
            link = url + l['href']
            # get content of a tag
            function_arguments.append(l)

    # print(f"Downloading {len(function_arguments)} files")

    print(function_arguments)

    # pool = multiprocessing.Pool(processes=2)
    # outputs = pool.map(download, function_arguments)
    
    for arg in function_arguments:
        download(arg)