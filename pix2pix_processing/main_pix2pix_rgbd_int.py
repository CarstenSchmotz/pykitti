import os
import cv2
from PIL import Image
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

def clear_and_create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def convert_to_8bit(input_folder, output_folder, is_depth=False):
    clear_and_create_folder(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            if is_depth:
                # Convert to grayscale if it's a depth map
                if img.mode != 'I':
                    img = img.convert('I')
                    print(f"{filename} converted to 32-bit grayscale mode for depth map.")

                print(f"{filename} is a depth map and will be processed.")

                # Convert the image to a numpy array
                img_array = np.array(img)

                # Normalize the depth values to the range [0, 255]
                img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
                img_array = img_array.astype(np.uint8)

                # Convert the numpy array back to an image
                img_8bit = Image.fromarray(img_array, 'L')
            else:
                # Convert image to RGB if it's not already in that mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print(f"{filename} converted to RGB mode.")

                print(f"{filename} is an image and will be processed.")

                # Convert the image to a numpy array and scale values
                img_array = np.array(img)
                img_array = img_array.astype(np.uint8)

                # Convert the numpy array back to an image
                img_8bit = Image.fromarray(img_array, 'RGB')

            # Save the 8-bit image to the output folder
            output_path = os.path.join(output_folder, filename)
            img_8bit.save(output_path)

            print(f"Processed {filename} and saved as 8-bit image.")
    return output_folder

def concatenate_images(folder1, folder2, output_folder):
    filenames1 = os.listdir(folder1)
    filenames2 = os.listdir(folder2)
    
    filenames1.sort()
    filenames2.sort()
    
    for filename1, filename2 in zip(filenames1, filenames2):
        image_path1 = os.path.join(folder1, filename1)
        image_path2 = os.path.join(folder2, filename2)
        
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if image1 is None or image2 is None:
            print(f"Error loading images: {image_path1}, {image_path2}")
            continue
        
        min_height = min(image1.shape[0], image2.shape[0])
        image1 = image1[:min_height, :]
        image2 = image2[:min_height, :]
        
        concatenated_image = cv2.hconcat([image1, cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)])
        
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename1)[0]}_{os.path.splitext(filename2)[0]}.jpg")
        cv2.imwrite(output_path, concatenated_image)
        
        print(f"Saved concatenated image at: {output_path}")
    return output_folder

def split_dataset(output_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    all_files = [f for f in os.listdir(output_folder) if os.path.isfile(os.path.join(output_folder, f))]
    
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"

    train_files, temp_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_ratio/(test_ratio + val_ratio), random_state=42)

    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for f in train_files:
        shutil.move(os.path.join(output_folder, f), os.path.join(train_folder, f))
    for f in val_files:
        shutil.move(os.path.join(output_folder, f), os.path.join(val_folder, f))
    for f in test_files:
        shutil.move(os.path.join(output_folder, f), os.path.join(test_folder, f))
    
    print(f"Dataset split into {len(train_files)} training, {len(val_files)} validation, and {len(test_files)} test images.")

if __name__ == "__main__":

    input_folder_rgb = "/Users/carstenschmotz/Downloads/image"# use from kitty
    input_folder_depth = "/Users/carstenschmotz/Downloads/test" #use depth from depthanoted
    bit_output_folder_rgbd = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_bitrgb"
    bit_output_folder_depth = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_bitdepth"
    output_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_pix2pix"
    lidar_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_folder_pointcloud"# use from kitty after safe pointcloud
    lidar_output = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_folder_pointcloud_8bit"
    input_folder_rgbd = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_folder_rgbd"
    clear_and_create_folder(bit_output_folder_rgbd)
    clear_and_create_folder(bit_output_folder_depth)
    clear_and_create_folder(output_folder)
    #clear_and_create_folder(lidar_output)

    folder_rgbd = convert_to_8bit(input_folder_rgbd, bit_output_folder_rgbd)
    folder_int = convert_to_8bit(lidar_folder, lidar_output)
    #folder_depth = convert_to_8bit(input_folder_depth, bit_output_folder_depth, is_depth=True)
    
    concatenate_images(bit_output_folder_rgbd, lidar_output, output_folder)
    split_dataset(output_folder)
