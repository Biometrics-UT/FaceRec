import time

import itertools
import json
import time
from typing import List
from scipy import fftpack
from pathlib import *
import numpy as np
import face_recognition
import dlib,imageio
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src.utils import resize_image_exact, compute_encoding, get_dict, get_best_encoding, remove_wrong_indexes
import cv2
import os
os.makedirs("images/fft",exist_ok=True)
number = np.random.randint(0,100)
def get_fft(original_path:Path,type,name:str):
    print(original_path.__str__())
    assert original_path.exists() and original_path.is_dir()
    t = time.time()
    original_files: List[Path] = [f for f in original_path.glob('**/*.' + type) if f.is_file()]
    file=original_files[number]
    print("Loading filename took:", time.time() - t)
    t = time.time()
    img = cv2.imread(str(file),0)

    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    fig = plt.gcf()
    fig.suptitle(name, fontsize=14)
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

    plt.savefig(f"images/fft/{name}-fft.png")
    plt.close()
    img = cv2.imread(str(file), 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back)
    plt.title('Result in JET'), plt.xticks([]), plt.yticks([])


    plt.savefig(f"images/fft/{name}-hpf.png")
    plt.close()
    image = imageio.imread(str(file), as_gray =True)
    fft2 = fftpack.fft2(image)

    plt.imshow(abs(fft2))
    plt.show()

names = ["original", "Deep Normal", "Deep Sobel", "Sobel", "Normal", "Deep Laplacian", "Laplacian"]

get_fft(original_path=Path("../data/originals"), type="jpg",name=names[0])
for i in range(1, 7):
    get_fft(original_path=Path("../data/improved" + str(i)), type="png",name=names[i])