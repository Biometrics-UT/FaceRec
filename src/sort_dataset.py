import time
from collections import defaultdict
from typing import List

from pathlib import *
from shutil import copyfile

original_path = Path("../data/originals")
assert original_path.exists() and original_path.is_dir()
t = time.time()
original_files: List[Path] = [f for f in original_path.glob('**/*.jpg') if f.is_file()]
d = defaultdict(list)
for file in original_files:
    d[file.name.split("d")[0]].append(file)
save_path = Path("../data/classified")
for k, v in d.items():
    folder_path = save_path / k
    folder_path.mkdir(parents=True, exist_ok=True)
    for img_path in v:
        copyfile(img_path.absolute().__str__(), (folder_path / img_path.name).absolute().__str__())
