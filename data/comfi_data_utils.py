import torch
import os
import torch.nn as nn
import matplotlib.pyplot as plt
import monai
import ants
import tarfile
import os, gzip, shutil
import scipy.ndimage as ndi
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity, Lambda, RandSpatialCrop
from monai.networks.nets import UNet
from torch.utils.data import random_split
import numpy as np

SIZE = (256, 256) # (Height, Width) of the generated bias field
num_points = 60 # resolution of the simulated coil array

# Uniform Random Variable Bounds
LOW_BOOST_BOUNDS = (0, 0.2)
COIL_VERT_POS_BOUNDS = (SIZE[0] - 1, SIZE[0] * 1.2)
PARAM_B_ADJUST_BOUNDS = (-0.02, 0.02)
COIL_WIDTH_BOUNDS = (0.1, 0.4)
BATCH_SIZE = 1

def gz_extract(directory):
    extension = ".gz"
    os.chdir(directory)
    for item in os.listdir(directory): # loop through items in dir
      if item.endswith(extension): # check for ".gz" extension
          gz_name = os.path.abspath(item) # get full path of files
          file_name = (os.path.basename(gz_name)).rsplit('.',1)[0] #get file name for file within
          with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
              shutil.copyfileobj(f_in, f_out)
          os.remove(gz_name) # delete zipped file

# Custom Function to fetch list of file names for DataLoader
def fetch_nii_file_names(dir_name):
    gz_extract(dir_name)
    filenames = []
    for item in os.listdir(dir_name): # loop through items in dir
        if item.endswith(".nii"): # check for ".nii" extension
            filenames.append(item)
    return filenames

# Custom Functions to add Rician noise
def clip_img(image):
  image = image.double()
  image = torch.where(image > 1., 1., image)
  return torch.where(image < 0., 0., image)

def add_rician_noise(image, intensity=1):
  n1 = torch.normal(0, 1, image.shape)
  n1 = n1 / torch.max(n1)
  n2 = torch.normal(0, 1, image.shape)
  n2 = n2 / torch.max(n2)
  return clip_img(torch.abs(image + intensity*n1 + intensity*n2*1j))

def add_gaussian_noise(image, mean=0.0, std=1.0, intensity=1.0):
  noise = torch.normal(mean=mean, std=std, size=image.shape)
  noise = noise
  noisy_image = image + intensity * noise
  return clip_img(noisy_image)

# Custom Functions for N4 Correction
def run_n4(img, mask):
  #img = img.copy()
  #mask = mask.copy()
  #img = img.detach().numpy()
  mask = mask.detach().numpy().astype('uint8')
  ants_img = ants.from_numpy((img * 255.).astype('uint8').T)
  ants_mask = ants.from_numpy(mask.T)
  ants_mask = ants_mask / ants_mask.max()
  ants_mask = ants_mask.threshold_image( 1, 2 )
  n4_corr = ants.n4_bias_field_correction(ants_img, mask=ants_mask, rescale_intensities=True)
  #n4_corr = ants.abp_n4(ants_img)
  return n4_corr.numpy().T / 255.

# Custom Functions to induce inhomogeneity
def poly_dec(x):
  return 1927.5 * (x + 37)**-2.093

def genBiasField(SIZE, coil_left, coil_right, coil_vert_pos, b_adj, low_boost):
  global a, b, c, d

  # Define Coil Shape
  cx = torch.linspace(round(SIZE[1]*coil_left), round(SIZE[1]*coil_right), steps=num_points) # Horizontal coordinates

  # Put coil array at or below bottom edge of bias field
  y_pos = round(coil_vert_pos)
  cy = torch.linspace(y_pos, y_pos, steps=num_points) # Vertical coordinates

  coils = torch.stack([cy, cx], axis=0).T # Reshape to prepare for arithmetic operations

  B = torch.zeros(SIZE)
  dists = torch.zeros((coils.shape[0],)) # Distances between coil points and field points

  # Exponential curve random perturbations
  #local_b = b + b_adj

  # Loop over all pixels in B
  for i in range(B.shape[0]):
    for j in range(B.shape[1]):
      # Stack of copies of this point's coordinates
      p = torch.tensor([i, j])
      p = torch.tile(p, (num_points, 1))

      # Get the distance between this point and the closest coil point
      dist = torch.min(torch.linalg.norm(coils - p, axis=1))

      # Simulate exponential falloff
      #B[i, j] = exp_dec(dist, a, local_b, c, d)
      B[i, j] = poly_dec(dist)
  
  # Normalize B on range [0, 1]
  B_norm = normalize(B)

  # Scale up / boost the weak end intensity of the field
  B_boosted = B_norm * (1 - low_boost) + low_boost

  return B_boosted

def genCompositeField(SIZE, lb):
  num_coils = 1 #random.randint(1, 3)

  coil_width = 0.3 #sampleRV(COIL_WIDTH_BOUNDS)
  if num_coils == 1:
    coil_width += 0.1
  coil_left_bound, coil_right_bound = 0.5 - coil_width, 0.5 + coil_width # horizontal extent of coil array
  sub_fields = torch.zeros((num_coils, SIZE[0], SIZE[1]))
  c_fraction = 1. / num_coils
  vert_pos = 350 #sampleRV(COIL_VERT_POS_BOUNDS)
  b_adj = 0 #sampleRV(PARAM_B_ADJUST_BOUNDS)
  low_boost = lb #sampleRV(LOW_BOOST_BOUNDS)

  for i in range(num_coils):
    sub_fields[i, :, :] = genBiasField(SIZE, coil_left_bound*c_fraction + i*c_fraction, coil_right_bound*c_fraction + i*c_fraction, vert_pos, b_adj, low_boost)

  return sub_fields.mean(axis=0)

def normalize(image):
  image = torch.tensor(image)
  new_image = image - torch.min(image)
  return new_image / torch.max(new_image)




def combinedTransforms(img, B):
  N_GAIN = 0.1 # Percent intensity of the magnitude of the absolute value of the noise  0.07
  #B_LOW_END = 0.055 # Percent intensity of the darkest part of the B field  0.06
  
  img = torch.tensor(img)
  mask_np = torch.where(torch.tensor(img) > 0, 1, 0) #.astype('float32')
  norm_Img = normalize(img*B*mask_np)
  #norm_Img = normalize(img*mask_np)
  #img = img*B*mask_np
  img_x = add_rician_noise(norm_Img, intensity=N_GAIN)
  img_n4 = img_x
  img_n4 = run_n4(img_x.detach().numpy(), mask_np)
  return torch.tensor(img_n4)



# Combined Image to be used for Training
def finalTransforms(img_3d, B):
    #print(img_3d.shape)
    z = img_3d.shape[0]
    #print(shape)
    #N_GAIN = 0.06 # Percent intensity of the magnitude of the absolute value of the noise  0.07
    #torch.random(0,5)
    # plt.imshow(B, cmap='gray')
    # plt.show()
    # plt.savefig("bias_pre.png")
    B = torch.rot90(torch.rot90(torch.rot90(B)))
    # plt.imshow(B, cmap='gray')
    # plt.show()
    # plt.savefig("bias_post.png")
    #plt.imshow(B, cmap='gray')
    for i in range(0,z):
        img_3d[i,:,:]=combinedTransforms(img_3d[i,:,:], B)
    #img_3D = combinedTransforms(img_3d, B)
    return img_3d



ip_train_transforms = Compose([
                             #ScaleIntensity(), 
                             EnsureChannelFirst(),
                             Lambda(finalTransforms), 
                             ])

label_train_transforms = Compose([  
                             ScaleIntensity(), 
                             EnsureChannelFirst(),
                             
                            ])