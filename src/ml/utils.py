from pathlib import Path
from typing import List


def download_file(url: str, path: str = "../../data/temp.tar") -> bool:
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError, URLError
    import sys
    if Path(path).exists() and Path(path).is_file():
        return False
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})  # Because it is less suspicious
        f = urlopen(req)
        with open(path, 'wb+') as local_file:
            local_file.write(f.read())
        return True
    except HTTPError as e:
        print(e)
        sys.exit(-1)
    except URLError as e:
        print(e)
        sys.exit(-1)


def check_md5(md5: str, path: str = "../../data/temp.tar"):
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest() == md5


def get_all_files(path: str) -> List[Path]:
    p = Path(path).glob('**/*')
    return [x for x in p if x.is_file()]


def extract_tar(path: str = "../../data/temp.tar", out_path: str = "../../data/dataset_train") -> str:
    import tarfile
    import shutil
    with tarfile.open(path) as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, out_path)
    for file in get_all_files(out_path):
        shutil.move(file.absolute().__str__(), out_path)
    dirs = [x for x in Path(out_path).iterdir() if not x.is_file()]
    for _dir in dirs:
        if _dir.is_dir():
            if any(_dir.iterdir()):
                print("Removing", list(_dir.iterdir()))
            shutil.rmtree(_dir)
        else:
            print("This is not a directory nor a file, please check it", dir)
    return path