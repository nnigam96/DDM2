from dipy.io.image import load_nifti
from dipy.data import get_fnames
from core.metrics import *
from matplotlib import pyplot as plt
from monai.transforms import ScaleIntensity 
from data.comfi_dataset import MRIDataset
from PIL import Image, ImageOps
import numpy as np
import torch
from piq import brisque
from piq import ssim
from piq import psnr
# qiyuan's data
from data.comfi_data_utils import *
valid_mask = [15,90]
#valid_mask = valid_mask.astype(np.bool8)
dataset = MRIDataset('/staging/nnigam/inphase/anatid_^_0003_Usab_HeadToe_^_3401_^_InPhase__MRAC_2_-_(2)_Head_to_Toe_Emission_ph.nii', valid_mask,
                     phase='train', val_volume_idx=60, padding=3)#, initial_stage_file='/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_noisemodel/stages.txt')

data = dataset[7500]
raw = data['raw']
raw = ScaleIntensity()(raw)
mask_np = torch.where(torch.tensor(raw) > 0, 1, 0) #.astype('float32')
dataset.B = torch.rot90(torch.rot90(torch.rot90(dataset.B)))

# Rotate image by 90 degrees counter-clockwis
raw = normalize(raw*dataset.B*mask_np)
raw = raw.detach().numpy()
raw = 2*raw - 1
dataset.B = 2*dataset.B - 1
img = data['X']
condition = data['condition']
n4_op = img.numpy()
condition = condition.numpy()


# Load an image
img = Image.open('/staging/nnigam/ddm2_experiments/comfi_rician_bias_6pc_noise_22pc_230516_133452/results/22/12000_1_denoised.png')

# Convert the image to grayscale and then to numpy array
img = ImageOps.grayscale(img)
img = np.array(img)

from piq import brisque


# Get the first third of the image vertically
#img = img[:int(img.shape[0]/3), :]

# Get the second third of the image vertically
img = img[int(img.shape[0]/3)+1:int(2*img.shape[0]/3), :]

# Get the second half of the image horizontally
img = img[:, int(img.shape[1]/2):]

brisque_score_final = brisque(torch.tensor(img).unsqueeze(0).unsqueeze(0), data_range=255.0)

# # Rotate images by 90 degrees counter-clockwise
# #img = np.rot90(np.rot90(img).squeeze())
# dataset.B = np.rot90(dataset.B)
# raw = np.rot90(raw)
img = img - img.min()
img = img/img.max()

img = 2*img - 1

# Flip the image vertically
#n4_op = np.flip(n4_op.squeeze(0), axis=0)

#n4_op = np.rot90(n4_op.squeeze(0), 2)
n4_op = n4_op.squeeze(0)
#dataset.B = 2*dataset.B - 1
vis = np.hstack((raw, dataset.B, n4_op, img[:-2, :-3]))

raw = raw +1
raw = raw/2
raw_brisque = brisque(torch.tensor(raw).unsqueeze(0).unsqueeze(0))
# Rotate every quarter of vis by 90 degrees counter-clockwise
vis[:, :int(vis.shape[1]/4)] = np.rot90(vis[:, :int(vis.shape[1]/4)])
vis[:, int(vis.shape[1]/4):int(vis.shape[1]/2)] = np.rot90(vis[:, int(vis.shape[1]/4):int(vis.shape[1]/2)])
vis[:, int(vis.shape[1]/2):int(3*vis.shape[1]/4)] = np.rot90(vis[:, int(vis.shape[1]/2):int(3*vis.shape[1]/4)])
vis[:, int(3*vis.shape[1]/4):] = np.rot90(vis[:, int(3*vis.shape[1]/4):])

plt.imshow(vis, cmap='gray')
plt.show()
plt.axis('off')
plt.savefig("dummy_name.png", transparent=True, pad_inches=0)

#img = ScaleIntensity()(img)
img = img + 1
img = img/2




def re_normalize(img):
     img = (img+1)/2
     return img
# denoised = Image.open('/home/nnigam/3D Unet Model.png')

'''
img = Image.open('/home/nnigam/3D Unet Model.png')
img = ImageOps.grayscale(img)
img = np.array(img)
# Divide the image into 3 parts based on length of the image
# img = img[:, :int(img.shape[1]/3)]
ip = torch.tensor(img[:, :int(img.shape[1]/3)]).unsqueeze(0).unsqueeze(0)
# Add white noise to the image where pixels are between 0 and 255
# def add_white_noise(image):
#     noise_img = image + torch.normal(0, 50, size=image.shape)
#     noise_img = torch.clip(noise_img, 0, 255)
#     noise_img = torch.tensor(255*noise_img/noise_img.max())
#     return noise_img
# ip = add_white_noise(image=ip)
n4_op = torch.tensor(img[:, int(img.shape[1]/3):int(2*img.shape[1]/3)]).unsqueeze(0).unsqueeze(0)
final_op = torch.tensor(img[:, int(2*img.shape[1]/3):-1]).unsqueeze(0).unsqueeze(0)
psnr_n4 = psnr(n4_op, ip, data_range=255)
psnr_final = psnr(final_op, ip, data_range=255)
# ssim_final = ssim(ip, op)
# ssim_denoised = ssim(ip, bias)
'''
print('Break')#break
