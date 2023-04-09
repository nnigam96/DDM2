from dipy.io.image import load_nifti
from dipy.data import get_fnames
from core.metrics import *
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
import numpy as np
import torch

raw_data, _ = load_nifti('D:/Comfi Data/inphase-004/inphase/anatid_^_0086_UBrain_^_601_^_InPhase__MRAC_1_-_Static_Brain_Emission_ph.nii')

raw_data = raw_data.astype(np.float32) / np.max(raw_data, axis=(0,1,2), keepdims=True)

raw_data = torch.tensor(raw_data, dtype=torch.float32)
raw_ip = raw_data[:,:,60].numpy()

denoised_image = Image.open("D:/Git Repos/DDM2/experiments/hardi150_denoise_230331_131313/results/0_1_denoised.png")
denoised_arr = np.asarray(ImageOps.grayscale(denoised_image))
denoised_arr = denoised_arr.astype(np.float32) / np.max(denoised_arr, axis=(0,1), keepdims=True)



noisy_img = Image.open("D:/Git Repos/DDM2/experiments/hardi150_denoise_230331_131313/results/0_1_input.png")
noisy_arr = np.asarray(ImageOps.grayscale(noisy_img))
noisy_arr = noisy_arr.astype(np.float32) / np.max(noisy_arr, axis=(0,1), keepdims=True)

vis = np.hstack((raw_ip, noisy_arr, denoised_arr))
#vis = np.hstack((img[0], condition[0], condition[1]))
plt.imshow(vis, cmap='gray')
plt.show()

psnr = calculate_psnr(255. * noisy_arr, 255 * denoised_arr)

print('break')
