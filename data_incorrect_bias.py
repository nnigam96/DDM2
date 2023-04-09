import torch
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, ScaleIntensity, Lambda, RandSpatialCrop
from monai.networks.nets import UNet
from torch.utils.data import random_split
from utils_incorrect_bias import *
#from constants import *



def combinedTransforms(img, B):
  N_GAIN = 0.06 # Percent intensity of the magnitude of the absolute value of the noise  0.07
  #B_LOW_END = 0.055 # Percent intensity of the darkest part of the B field  0.06
  mask_np = torch.where(img > 0, 1, 0).astype('float32')
  norm_Img = normalize(img*B*mask_np)
  #img = img*B*mask_np
  img_x = add_rician_noise(norm_Img, N_GAIN)
  #img_n4 = run_n4(img_x.detach().numpy(), mask_np)
  return torch.tensor(img_x)

# Combined Image to be used for Training
def finalTransforms(img_3d):
    #print(img_3d.shape)
    z = img_3d.shape[3]
    #print(shape)
    #N_GAIN = 0.06 # Percent intensity of the magnitude of the absolute value of the noise  0.07
    B_LOW_END = 0.055 # Percent intensity of the darkest part of the B field  0.06
    B = genCompositeField((256, 256), B_LOW_END)
    #torch.random(0,5)
    B = torch.rot90(torch.rot90(torch.rot90(B)))
    #plt.imshow(B, cmap='gray')
    for i in range(0,z):
        img_3d[:,:,:,i]=combinedTransforms(img_3d[:,:,:,i], B)
    return img_3d



ip_train_transforms = Compose([# Resize((256,256,120)),
                             #Lambda(print_val),
                             ScaleIntensity(), 
                             #Lambda(print_val),
                             EnsureChannelFirst(),
                             #Lambda(print_val),
                             Lambda(finalTransforms), 
                             RandSpatialCrop(roi_size=[256, 256, 32], random_size=False)]
                             )

label_train_transforms = Compose([ #Resize((256,256,120)),
                             ScaleIntensity(), 
                             EnsureChannelFirst(),
                             RandSpatialCrop(roi_size=[256, 256, 32], random_size=False)

                            ])

#def get_loaders(dataset):    
#    train_size = round(0.5*len(dataset))
#    val_size = round(0.3*len(dataset))
#    test_size = len(dataset) - train_size - val_size
#
#    train, test, val  = random_split(dataset,[train_size, val_size, test_size])
#    
#    train_loader = DataLoader(train, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available())
#    test_loader = DataLoader(test, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available(), shuffle = True)
#    val_loader = DataLoader(val, batch_size=BATCH_SIZE, pin_memory=torch.cuda.is_available())
#
#    return train_loader, test_loader, val_loader
    
