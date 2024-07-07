import os
import cv2
from PIL import Image
import numpy as np
import shutil

def clear_and_create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# Function to concatenate images from two folders and save them
def concatenate_images(folder1, folder2, output_folder):
    filenames1 = os.listdir(folder1)
    filenames2 = os.listdir(folder2)
    
    # Sort filenames to ensure they match pairwise
    filenames1.sort()
    filenames2.sort()
    
    # Iterate over filenames and concatenate images
    for filename1, filename2 in zip(filenames1, filenames2):
        image_path1 = os.path.join(folder1, filename1)
        image_path2 = os.path.join(folder2, filename2)
        
        # Load images
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        
        if image1 is None or image2 is None:
            print(f"Error loading images: {image_path1}, {image_path2}")
            continue
        
        # Ensure images have the same height
        min_height = min(image1.shape[0], image2.shape[0])
        image1 = image1[:min_height, :]
        image2 = image2[:min_height, :]
        
        # Concatenate images horizontally
        concatenated_image = cv2.hconcat([image1, image2])
        
        # Save the concatenated image
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename1)[0]}_{os.path.splitext(filename2)[0]}.jpg")
        cv2.imwrite(output_path, concatenated_image)
        
        print(f"Saved concatenated image at: {output_path}")
    return output_folder

# Function to convert images in a folder to 8-bit
"""def convert_to_8bit(input_folder, output_folder):
    

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            if img.mode == 'RGB':
                print(f"{filename} is a 24-bit RGB image and will be processed.")
                
                img_array = np.array(img)
                img_array = img_array.astype(np.uint8)
                img_8bit = Image.fromarray(img_array, 'RGB')
                
                output_path = os.path.join(output_folder, filename)
                img_8bit.save(output_path)
                
                print(f"Processed {filename} and saved as 8-bit RGB image.")
            else:
                print(f"{filename} is not a 24-bit RGB image and will be skipped.")
    return output_folder
    """
def convert_to_8bit(input_folder, output_folder):
    clear_and_create_folder(output_folder)
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

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

            print(f"Processed {filename} and saved as 8-bit RGB image.")
    return output_folder


# Main block
if __name__ == "__main__":
    input_folder_rgb = "/Users/carstenschmotz/Downloads/image"
    input_folder_depth = "/Users/carstenschmotz/Downloads/test"
    bit_output_folder_rgb = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_bitrgb"
    bit_output_folder_depth = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_bitdepth"
    output_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output"

    # Ensure output folders are created and cleared
    clear_and_create_folder(bit_output_folder_rgb)
    clear_and_create_folder(bit_output_folder_depth)
    clear_and_create_folder(output_folder)

    folder_rgb = convert_to_8bit(input_folder_rgb, bit_output_folder_rgb)
    folder_depth = convert_to_8bit(input_folder_depth, bit_output_folder_depth)
    
    concatenate_images(folder_rgb, folder_depth, output_folder)
