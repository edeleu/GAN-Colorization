# COS429 cWGAN Image Colorization Project

# Authors: Edward Deleu and Vishva Ilavelan 
# Note: attributions for specific segments of code will be given throughout this file.
# Also, data analysis, processing, and visualization are performed in separate files.
# ie, figure-producing, graphing, computing error, and creating heatmaps. 
# these are not attached to the submission to reduce clutter, but may be provided at request

# begin with MANY import statements!
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import PIL
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import matplotlib.image

import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torch.utils.data import random_split
import gc
import torch

# from torchsummary import summary

# move to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is ", device)

# Stage 0: Helper functions

# Define a function to convert normalized LAB colorspace, as produced by the generator, back to RGB
# Method inspired by https://www.kaggle.com/code/varunnagpalspyz/pix2pix-is-all-you-need/notebook
def lab_to_rgb(L, a, b):
    """
    Takes a single image and converts from LAB space to RGB space.
    """
    # re-map the normalized LAB values to actual LAB values
    L = (L + 1.) * 50
    a = a * 128
    b = b * 128

    # move data to CPU and detach from computational graph
    Lab = torch.cat([L, a, b], dim=0).detach().cpu().numpy() # CxHxW
    Lab = np.moveaxis(Lab,0,-1)  # Now H*W*C
    img_rgb = lab2rgb(Lab)
   # print(img_rgb.shape)
    return img_rgb

# Stage 1: Create a dataset class for use with the PyTorch training process:)
# Implementation is our own.
# Returns L, a, b channels in the CxHxW format.
class ColorizationDataset(Dataset):
    ''' Black and White (L) Images and corresponding A&B Colors'''
    def __init__(self, imagePaths, transform=None):
        '''
        :param dataset: Dataset name.
        :param data_dir: Directory with all the images.
        :param transform: Optional transform to be applied on sample
        '''
        self.paths = imagePaths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # img is shape (600,800) or WxH

        if self.transform:
            img = self.transform(img)

        #img is now shape (3,800,600)->(3,384,288) or CxHxW
        img = np.array(img)

        #Scikit learn RGB2LAB expects channel dimension at end instead of beginning, swap so we have (288,384,)
        img = np.moveaxis(img,0,-1) 

        lab = rgb2lab(img).astype("float32")
    
        lab = transforms.ToTensor()(lab)

        # Normalize data to -1 to 1 using the fact that: L in [0,100] and ab in [-128,127]
        L = lab[[0],...]/50 - 1
        a = lab[[1],...]/128
        b = lab[[2],...]/128
        # L, a, b are each (1, 384, 288) CxHxW

        # ensure LAB conversion may be reversed properly (test case)
        # RGB = lab_to_rgb(L,a,b)
        # matplotlib.image.imsave("/scratch/network/vi6908/Colorization_GAN/training_photos/IMAGE.jpg", RGB)

        return L, a, b

## DEFINE GENERATOR STRUCTURE using Residual U-Net
# Code modified from https://www.kaggle.com/code/salimhammadi07/pix2pix-image-colorization-with-conditional-wgan/notebook 

# Define residual blocks for use in the U-Net structure
# residual block contains a residual skip connection
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,normalization=True):
        super().__init__()
        if normalization:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, stride=stride, bias=False),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels,kernel_size=3,padding=1, stride=stride, bias=False),
                nn.ReLU()
            )

        # use 1x1 conv. for identity mapping of the channel dimension
        self.identity_map = nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=stride)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        # x = inputs.clone().detach()
        out = self.layer(inputs)
        residual  = self.identity_map(inputs)
        skip = out + residual
        return self.relu(skip)

# Define a downsampling block for the encoder using the residual block
class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)

# Define an upsampling block for the decoder section
class UpSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        #Swap transpose convolution in for slightly higher model capacity
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

# Put the building blocks together to create the actual generator!
# code still modified from 
# https://www.kaggle.com/code/salimhammadi07/pix2pix-image-colorization-with-conditional-wgan/notebook
# with updated data handling and some architectural tweaks
class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, dropout_rate = 0.2):
        super().__init__()
        self.encoding_layer1_ = ResBlock(input_channel,64,normalization=False)
        self.encoding_layer2_ = DownSampleConv(64, 128)
        self.encoding_layer3_ = DownSampleConv(128, 256)
        self.bridge = DownSampleConv(256, 512)
        self.decoding_layer3_ = UpSampleConv(512, 256)
        self.decoding_layer2_ = UpSampleConv(256, 128)
        self.decoding_layer1_ = UpSampleConv(128, 64)
        self.output = nn.Conv2d(64, output_channel, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, inputs):
        ###################### Enocoder #########################
        e1 = self.encoding_layer1_(inputs)
        e1 = self.dropout(e1)
        e2 = self.encoding_layer2_(e1)
        e2 = self.dropout(e2)
        e3 = self.encoding_layer3_(e2)
        e3 = self.dropout(e3)
        
        ###################### Bridge #########################
        bridge = self.bridge(e3)
        bridge = self.dropout(bridge)
        
        ###################### Decoder #########################
        d3 = self.decoding_layer3_(bridge, e3)
        d2 = self.decoding_layer2_(d3, e2)
        d1 = self.decoding_layer1_(d2, e1)
        
        ###################### Output #########################
        out = self.output(d1)
        outputA, outputB = torch.split(out, 1, dim=1)
        return outputA, outputB

## NEXT, DEFINE THE DISCRIMINATOR (CRITIC)!  This structure is 
# also modified from https://www.kaggle.com/code/salimhammadi07/pix2pix-image-colorization-with-conditional-wgan/notebook
# with data and architectural tweaks for improved performance

class Critic(nn.Module):
    def __init__(self, in_channels=3):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters,normalization=True):
            """Returns layers of each critic block"""
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *critic_block(in_channels, 64, normalization=False),
            *critic_block(64, 128),
            *critic_block(128, 256),
            *critic_block(256, 512),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, l, a, b):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((l, a, b), 1)
        output = self.model(img_input)
        return output

# Define a weight-initialization method to improve GAN learning process
# function adapted from tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.normal_(m.weight.data, 0.0, 0.02)

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Now, define a training function that accepts a data loader, generator, and critic.
# this is custom-written for our task, although, the WGAN training loss implementation is adapted from 
# https://github.com/jalola/improved-wgan-pytorch and https://github.com/eriklindernoren/PyTorch-GAN

def train(train_loader, generator, critic):

    optimizer_Generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0, 0.9)) 
    optimizer_Critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0, 0.9))
    #optimizer_Generator = optim.RMSprop(generator.parameters(), lr=learning_rate)
    #optimizer_Critic = optim.RMSprop(critic.parameters(), lr=learning_rate)

    L1_Criterion = nn.L1Loss()

    batches_done = 0
    for epoch in range(N_EPOCHS):
        for i, (L, a, b) in enumerate(train_loader):
            gc.collect()
            torch.cuda.empty_cache()
            L = L.to(device)
            a = a.to(device)
            b = b.to(device)

            # L, a, b are shape 16,1,384,288 or BxCxHxW

            #  Train Discriminator

            optimizer_Critic.zero_grad()
            
            # Ensure critic parameters are active
            for p in critic.parameters():
                p.requires_grad = True 

            # Generate a batch of images. Since we're training the Discriminator, detach Generator outputs so that parameters aren't updated
            fake_a, fake_b = generator(L)

            fake_a = fake_a.detach()
            fake_b= fake_b.detach()

            # Adversarial loss, = -mean(discriminator(real_imgs)) + mean(discriminator(fake_imgs))
            real_Loss_D = torch.mean(critic(L, a, b))
            fake_Loss_D = torch.mean(critic(L, fake_a, fake_b))
            loss_D = -real_Loss_D + fake_Loss_D

            loss_D.backward()
            optimizer_Critic.step()

            # Clip weights of discriminator
            for p in critic.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            # Train the generator every n_critic iterations
            if i % n_critic_iters == 0:

                #  Train Generator

                # Ensure critic parameters are frozen
                for p in critic.parameters():
                    p.requires_grad = False 

                optimizer_Generator.zero_grad()

                # Generate a batch of images, now connected in the autograd graph
                genA, genB = generator(L)
                # Adversarial loss on generated images
                loss_G = -torch.mean(critic(L, genA, genB))
                #loss_G.backward()

                # add reconstruction loss to generator loss as discussed in the pix2pix paper.
                reconstruction_loss = torch.mean((L1_Criterion(genA, a) + L1_Criterion(genB, b)))
                totalGenloss = loss_G + GEN_LAMBDA*reconstruction_loss

                totalGenloss.backward()
                optimizer_Generator.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G_GAN loss: %f] [G_L1 loss: %f] [G Total loss: %f]"
                    % (epoch, N_EPOCHS, batches_done % len(train_loader), len(train_loader), loss_D.item(), loss_G.item(),reconstruction_loss.item(),totalGenloss.item()), flush=True
                )
                
            batches_done += 1
        
            # save the first image of the last batch in the training data
            if (i == len(train_loader)-1):
                generatedA, generatedB = generator(L)
                currentImageFake = lab_to_rgb(L[0],generatedA[0],generatedB[0])
                currentImageReal = lab_to_rgb(L[0],a[0],b[0])

                #localDir = "/scratch/network/vi6908/Colorization_GAN/training_photos/"
                localDir = "/scratch/gpfs/ed5754/Colorization/modified/"
                strReal = str(localDir) + "Epoch_" + str(epoch) + "_" + str(i) + "Real.jpg"
                strFake = str(localDir) + "Epoch_" + str(epoch) + "_" + str(i) + "Fake.jpg"

                matplotlib.image.imsave(strReal, currentImageReal)
                matplotlib.image.imsave(strFake, currentImageFake)

                # if above a certain epoch threshold, save the model at this point as well.
                if (epoch>30):
                    strCritic = str(localDir) + "Epoch_" + str(epoch) + "_" + str(i) + "Critic.pt"
                    torch.save(critic.state_dict(), strCritic)

                    strGen = str(localDir) + "Epoch_" + str(epoch) + "_" + str(i) + "Gen.pt"
                    torch.save(generator.state_dict(), strGen)

# define an inference function for inferencing with a saved model
# accepts a data loader, generator and critic (not used)
def inference(inference_loader, generator, critic):
    generator.eval()
    # use eval() mode to turn off dropout for improved results

    # iterate thru batches of data, as normally
    for i, (L, a, b) in enumerate(inference_loader):
        # move data to GPU and generate a fake image
        #then, save real, fake, and grayscale variants
        print(i)
        L = L.to(device)
        a = a.to(device)
        b = b.to(device)

        generatedA, generatedB = generator(L)
        currentImageFake = lab_to_rgb(L[0],generatedA[0],generatedB[0])
        currentImageReal = lab_to_rgb(L[0],a[0],b[0])
        generatedA[0]=0
        grayImage = lab_to_rgb(L[0],generatedA[0],generatedA[0])

        # specify directory for inference run
        #localDir = "/scratch/network/vi6908/Colorization_GAN/training_photos/"
        localDir = "/scratch/gpfs/ed5754/Colorization/Output/Model_Instance/SEAsian/"
        strReal = str(localDir) + "Inference" + str(i) + "Real.jpg"
        strFake = str(localDir) + "Inference" + str(i) + "Fake.jpg"
        strGray = str(localDir) + "Inference" + str(i) + "Gray.jpg"

        matplotlib.image.imsave(strReal, currentImageReal)
        matplotlib.image.imsave(strFake, currentImageFake)
        matplotlib.image.imsave(strGray, grayImage)

        # limit to 300 photos per trial
        if (i>300):
            break

# main loop!
# here, we can swap function calls for varying functionality
if __name__=="__main__":    
    # example code of how to use torch.summary to get model information
    # model = Generator(1,2).to(device)
    # summary(model, (1, 288, 384), batch_size = 1)

    # crit = Critic(3).to(device)
    # summary(crit, [(1, 288, 384),(1, 288, 384), (1, 288, 384)], batch_size = 1)

    ## DEFINE TRAINING SETUP
    # set the same seed (for reproducability)
    torch.manual_seed(720) #old 720
    np.random.seed(720) #old720

    N_EPOCHS = 200
    learning_rate = 4e-5 # perhaps to 4e-5 ok, lower end
    batch_size = 8
    CLIP_VALUE = 0.01
    n_critic_iters = 5
    GEN_LAMBDA = 10
    
    # take advantage of A100 tensor cores
    torch.set_float32_matmul_precision('high')

    PROPORTION_TRAINING_DATA = 0.25 #old 0.25

    # load datasets
    rootDatasetPath = Path("/scratch/gpfs/ed5754/Colorization/")
    image_paths = list(rootDatasetPath.glob("clip_img/*/*/*"))
    trainingTransform = transforms.Compose([transforms.Resize((384,384)),transforms.RandomHorizontalFlip(),transforms.ToTensor()])

    full_dataset = ColorizationDataset(imagePaths=image_paths, transform=trainingTransform)
    train_size = int(PROPORTION_TRAINING_DATA * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # train dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory = True,num_workers=4,persistent_workers=True)
    # test dataloader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory = True, num_workers=4)

    ## CODE TO TRAIN FROM SCRATCH
    # generator = Generator(1,2).to(device)
    # generator = torch.compile(generator)
    # critic = Critic(3).to(device)
    # critic=torch.compile(critic)

    # generator.apply(weights_init)
    # critic.apply(weights_init)
    # train(train_loader,generator,critic)

    #FairFACE INFERENCE! Load dataset... 
    infer_paths = list(rootDatasetPath.glob("FairFace/train/Southeast_Asian/*"))
    infer_data = ColorizationDataset(imagePaths=infer_paths, transform=trainingTransform)
    infer_loader = DataLoader(infer_data, batch_size=1, shuffle=False, pin_memory = True, num_workers=4)

    # Code to load model from previous checkpoint! Used for inference
    modelPath = "/scratch/gpfs/ed5754/Colorization/BestGen.pt"
    criticPath = "/scratch/gpfs/ed5754/Colorization/EpochRes_75_1075Critic.pt"

    generator = Generator(1,2).to(device)
    generator = torch.compile(generator)
    generator.load_state_dict(torch.load(modelPath, weights_only=True))

    critic = Critic(3).to(device)
    critic=torch.compile(critic)
    critic.load_state_dict(torch.load(criticPath, weights_only=True))

    # call inference
    # inference(infer_loader,generator,critic)