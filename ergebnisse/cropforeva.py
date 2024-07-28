from PIL import Image
import os

def crop_image(input_image_path, output_folder):
    # Open the input image
    with Image.open(input_image_path) as img:
        img_width, img_height = img.size
        
        # Ensure the image is 1216x352
        if img_width != 1216 or img_height != 352:
            raise ValueError(f"Input image must be 1216x352, but is {img_width}x{img_height}")

        # Set the crop size
        crop_width, crop_height = 256, 256
        
        # Calculate the number of crops in each dimension
        crops_per_row = img_width // crop_width
        crops_per_col = img_height // crop_height
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Perform cropping
        crop_number = 1
        for row in range(crops_per_col):
            for col in range(crops_per_row):
                left = col * crop_width
                upper = row * crop_height
                right = left + crop_width
                lower = upper + crop_height
                
                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
                
                # Save the cropped image
                cropped_img_path = os.path.join(output_folder, f"crop_{crop_number}.png")
                cropped_img.save(cropped_img_path)
                crop_number += 1

# Example usage
input_image_path = r"D:\projekt_depth\for_training\depthv2\0000000011.png"
output_folder = r"D:\Dokumente\01_BA_Git\pykitti\ergebnisse"
crop_image(input_image_path, output_folder)
