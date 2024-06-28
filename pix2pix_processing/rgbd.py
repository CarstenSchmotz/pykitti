import os
import cv2
import numpy as np

def get_number_from_filename(filename):
    # Extract the number from the filename using regex
    match = re.search(r'\d+', filename)
    if match:
        return match.group(0)
    return None

def get_matching_files(folder1, folder2):
    files1 = os.listdir(folder1)
    files2 = os.listdir(folder2)

    # Create a dictionary with numbers from filenames as keys
    files_dict1 = {get_number_from_filename(f): f for f in files1 if get_number_from_filename(f)}
    files_dict2 = {get_number_from_filename(f): f for f in files2 if get_number_from_filename(f)}

    # Find common keys (numbers)
    common_keys = set(files_dict1.keys()).intersection(set(files_dict2.keys()))

    # Get the matching files
    matching_files = [(files_dict1[key], files_dict2[key]) for key in common_keys]

    return matching_files

def overlay_images(rgb_image, depth_image):
    # Ensure both images are the same size
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Normalize the depth image to range [0, 255] if it's not already
    if depth_image.dtype != np.uint8:
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Convert depth image to a 3-channel image
    depth_image_color = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    # Overlay the depth image onto the RGB image (blend with alpha)
    alpha = 0.6  # Transparency factor
    overlayed_image = cv2.addWeighted(rgb_image, alpha, depth_image_color, 1 - alpha, 0)

    return overlayed_image

def main(folder1, folder2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    matching_files = get_matching_files(folder1, folder2)

    for file1, file2 in matching_files:
        rgb_image_path = os.path.join(folder1, file1)
        depth_image_path = os.path.join(folder2, file2)

        rgb_image = cv2.imread(rgb_image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

        overlayed_image = overlay_images(rgb_image, depth_image)
        output_path = os.path.join(output_folder, f'overlayed_{file1}')
        cv2.imwrite(output_path, overlayed_image)

        print(f"Saved overlayed image: {output_path}")

if __name__ == "__main__":
    folder1 = 'path/to/rgb_folder'  # Path to the folder with RGB images
    folder2 = 'path/to/depth_folder'  # Path to the folder with depth images
    output_folder = 'path/to/output_folder'  # Path to the folder where overlayed images will be saved

    main(folder1, folder2, output_folder)
