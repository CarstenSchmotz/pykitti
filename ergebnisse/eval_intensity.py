import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim


predicted_path  = "/home/oq55olys/Projects/NN/pytorch-CycleGAN-and-pix2pix/results/pix2pix_kitti/val_latest/images/"
base_path = "/media/oq55olys/chonk/Datasets/kittilike/kitti-step/val/"
target_path = "/media/oq55olys/chonk/Datasets/kittilike/kitti-step/valB/"
merged_path = "/media/oq55olys/chonk/Datasets/kittilike/kitti-step/predicted/"


if not os.path.exists(merged_path):
    os.makedirs(merged_path)


def compute_metrics(img_in, img_tar):
    diff = np.abs(img_in.astype(np.float32) - img_tar.astype(np.float32))/255.0
    mse =diff**2
    #only keep diff values where img_tar is not zero
    diff_score = diff[np.where(img_tar != 0)]
    mse = mse[np.where(img_tar != 0)]
    mse = np.mean(mse)
    psnr = 10*np.log10(255*255/mse)
    l1 = np.mean(diff_score)

    #todo compare with mask
    ssim_full, ssim_img = ssim(img_in, img_tar, data_range= img_in.max() - img_in.min(), gaussian_weights=True, sigma=0.25, full=True)
    ssim_score = ssim_img[np.where(img_tar != 0)]
    ssim_masked = np.mean(ssim_score)
    #invert ssim_img
    #ssim_img = 255-ssim_img*255

    return l1, mse, psnr, ssim_masked



def read_image(path, num_parts, gray=True):

    stitch = num_parts > 1
    file_name = path.split('/')[-1]

    if stitch: 
        if not gray:
            print ('stitching only supports grayscale images')
            exit()
        #scale img_width to 4 times
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        merged = img
        img = cv2.resize(img, (img.shape[1]*4, img.shape[0]))
  
        prefix = file_name.split('_')[0] + '_' + file_name.split('_')[1]
        
        for j in range(3):
            path_next = path.replace('_0.png', '_'+str(j+1)+'.png')
            img = cv2.imread(path_next, cv2.IMREAD_GRAYSCALE)
            merged = np.hstack((merged, img))


        img = merged

    else:
        if gray:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
        #crop inner 1024x256
        w = img.shape[1]
        l = (w - 1024)//2
        r = l + 1024
        b = img.shape[0] - 256
        t = img.shape[0]

        img = img[b:t, l:r]
        prefix = path

    return img, prefix


predicted_files = os.listdir(predicted_path)
#sort
predicted_files.sort()
#load first file to get shape
gt_files = os.listdir(target_path)
gt_files.sort()

num_parts = 4

l1 = 0
num_files = len(predicted_files)
psnr= 0
ssim_score = 0
mse = 0

num_images = num_files//num_parts

num_files = num_images * num_parts

for i in range(0, num_files, num_parts):
    #read projected as grayscale

    merged, name = read_image(predicted_path+ predicted_files[i], num_parts)
    #make merged random grayscale values

    #get folder from kitti step which is the first part e.g 0013 and the file name without the folder
    dir = name.split('_')[0]
    og_name = name.split('_')[1]

    
    input_path = base_path + dir + '/image_02/data/'


    projected = read_image(target_path+ gt_files[i], num_parts)[0]
    merged_masked = merged.copy()
    merged_masked[projected == 0] = 0
    #stack projected and merged merged_masked vertically
    rgb= read_image(input_path+ og_name + '.png', 1, gray=False)[0]

    merged = np.vstack((merged, merged_masked, projected))
    merged = np.dstack((merged, merged, merged))
    merged = np.vstack((rgb, merged))

    l1_single, mse_single, psnr_single, ssim_score_single = compute_metrics(merged_masked, projected)

    l1 += l1_single
    mse += mse_single
    psnr += psnr_single
    ssim_score += ssim_score_single

    if i % 100 == 0:
        print(i, " out of ", num_files)
        cv2.imwrite(merged_path + name + '.png', merged)
        print('saved ' + merged_path + '/' + name + '.png')
        #print all losses for this file
        print("L1: ", l1_single, "PSNR: ", psnr_single, "SSIM: ", ssim_score_single, "MSE: ", mse_single)

l1 = l1/num_images
psnr = psnr/num_images
ssim = ssim_score/num_images
mse = mse/num_images
print("L1: ", l1, "PSNR: ", psnr, "SSIM: ", ssim, "MSE: ", mse)

exit()

