import os
from PIL import Image
import numpy as np

def convert_to_8bit(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open the image
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # Convert the image to a numpy array
            img_array = np.array(img)

            # Check the image mode and handle accordingly
            if img_array.dtype == np.uint16:
                # Scale the values to 8-bit range
                img_array = (img_array / 256).astype(np.uint8)

                # Convert the numpy array back to an image
                img_8bit = Image.fromarray(img_array)

                # Save the 8-bit image to the output folder
                output_path = os.path.join(output_folder, filename)
                img_8bit.save(output_path)

                print(f"Converted {filename} to 8-bit")
            else:
                print(f"{filename} is not a 16-bit image and will be skipped.")

# Example usage:
input_folder = r"C:\Users\Besitzer\Desktop\pix2pix\groundtruth_depth"  # Eingabeordner
output_folder = r"C:\Users\Besitzer\Desktop\pix2pix\output_bitrgb"  # Ausgabeordner
convert_to_8bit(input_folder, output_folder)
