import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Paths
predicted_path = r"D:\projekt_depth\for_training\eva\rgb" #rgb
base_path = r"D:\projekt_depth\results-20240727T161841Z-001\results\rgb\test_latest\images"
target_path =  r"D:\projekt_depth\for_training\eva\lidar" #groundtruth
merged_path =  r"D:\projekt_depth\for_training\eva\result"

if not os.path.exists(merged_path):
    os.makedirs(merged_path)

def compute_metrics(img_in, img_tar):
    diff = np.abs(img_in.astype(np.float32) - img_tar.astype(np.float32)) / 255.0
    mse = diff**2
    diff_score = diff[np.where(img_tar != 0)]
    mse = mse[np.where(img_tar != 0)]
    mse = np.mean(mse)
    psnr = 10 * np.log10(255 * 255 / mse)
    l1 = np.mean(diff_score)

    ssim_full, ssim_img = ssim(img_in, img_tar, data_range=img_in.max() - img_in.min(), gaussian_weights=True, sigma=0.25, full=True)
    ssim_score = ssim_img[np.where(img_tar != 0)]
    ssim_masked = np.mean(ssim_score)

    return l1, mse, psnr, ssim_masked

def read_image(path, num_parts, gray=True):
    if not os.path.isfile(path):
        print(f"File does not exist: {path}")
        return None, None

    stitch = num_parts > 1
    file_name = os.path.basename(path)

    if stitch:
        if not gray:
            print('Stitching only supports grayscale images')
            return None, None

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read image: {path}")
            return None, None

        merged = img
        img = cv2.resize(img, (img.shape[1] * 4, img.shape[0]))

        prefix = '_'.join(file_name.split('_')[:2])

        for j in range(3):
            path_next = path.replace('_0.png', f'_{j + 1}.png')
            img_next = cv2.imread(path_next, cv2.IMREAD_GRAYSCALE)
            if img_next is None:
                print(f"Failed to read image: {path_next}")
                return None, None
            merged = np.hstack((merged, img_next))

        img = merged
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if gray else cv2.imread(path)
        if img is None:
            print(f"Failed to read image: {path}")
            return None, None

        w = img.shape[1]
        l = (w - 1024) // 2
        r = l + 1024
        b = img.shape[0] - 256
        t = img.shape[0]

        img = img[b:t, l:r]
        prefix = path

    return img, prefix

predicted_files = sorted(os.listdir(predicted_path))
gt_files = sorted(os.listdir(target_path))

num_parts = 4

l1 = 0
psnr = 0
ssim_score = 0
mse = 0

num_files = len(predicted_files)
num_images = num_files // num_parts
num_files = num_images * num_parts

for i in range(0, num_files, num_parts):
    predicted_file = os.path.join(predicted_path, predicted_files[i])
    merged, name = read_image(predicted_file, num_parts)
    if merged is None:
        continue

    dir_name = name.split('_')[0]
    og_name = name.split('_')[1]

    input_path = os.path.join(base_path, dir_name, 'image_02', 'data')

    gt_file = os.path.join(target_path, gt_files[i])
    projected, _ = read_image(gt_file, num_parts)
    if projected is None:
        continue

    merged_masked = merged.copy()
    merged_masked[projected == 0] = 0

    rgb_file = os.path.join(input_path, og_name + '.png')
    rgb, _ = read_image(rgb_file, 1, gray=False)
    if rgb is None:
        continue

    merged = np.vstack((merged, merged_masked, projected))
    merged = np.dstack((merged, merged, merged))
    merged = np.vstack((rgb, merged))

    l1_single, mse_single, psnr_single, ssim_score_single = compute_metrics(merged_masked, projected)

    l1 += l1_single
    mse += mse_single
    psnr += psnr_single
    ssim_score += ssim_score_single

    if i % 100 == 0:
        print(f"{i} out of {num_files}")
        merged_file_path = os.path.join(merged_path, name + '.png')
        cv2.imwrite(merged_file_path, merged)
        print(f"Saved {merged_file_path}")
        print(f"L1: {l1_single} PSNR: {psnr_single} SSIM: {ssim_score_single} MSE: {mse_single}")

l1 /= num_images
psnr /= num_images
ssim = ssim_score / num_images
mse /= num_images
print(f"L1: {l1} PSNR: {psnr} SSIM: {ssim} MSE: {mse}")

exit()
