from skimage.metrics import structural_similarity as ssim_func
import numpy as np
import imageio
from torchvision import transforms
import cv2

ssim=[]
img = np.array(cv2.imread(f"../../../../lena.png"))
rec = np.array(cv2.imread(f"./fp_reconstruction_15.png"))
print(ssim_func(img, rec, channel_axis=-1))
