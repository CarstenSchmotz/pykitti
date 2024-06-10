from PIL import Image
import os

def convert_to_8bit(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # Open the image
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            
            # Convert the image to 8-bit
            img_8bit = img.convert("L")
            
            # Save the 8-bit image to the output folder
            output_path = os.path.join(output_folder, filename)
            img_8bit.save(output_path)

            print(f"Converted {filename} to 8-bit")

# Example usage:
input_folder = r"C:\Users\Besitzer\Desktop\pix2pix\output"#"D:\Dokumente\01_BA_Git\pykitti\outputforpix2pix"
output_folder = r"C:\Users\Besitzer\Desktop\pix2pix\output_bit"#"D:\Dokumente\01_BA_Git\pykitti\outputbit"
convert_to_8bit(input_folder, output_folder)
