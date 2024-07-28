from PIL import Image

def invert_black_to_blue(image_path, output_path):
    # Open the image
    img = Image.open(image_path)
    # Convert the image to RGBA (if not already in that mode)
    img = img.convert("RGBA")
    
    # Get the pixel data
    pixels = img.load()

    # Define the color to change black pixels to (blue)
    blue_color = (0, 0, 255, 255)  # RGBA for blue with full opacity

    # Loop through the image
    for y in range(img.height):
        for x in range(img.width):
            # Check if the pixel is black
            if pixels[x, y] == (0, 0, 0, 255):  # Fully opaque black
                # Change black pixel to blue
                pixels[x, y] = blue_color

    # Save the modified image
    img.save(output_path)
    print(f"Image saved to {output_path}")

