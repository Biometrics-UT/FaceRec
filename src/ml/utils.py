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
        tar.extractall(out_path)
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