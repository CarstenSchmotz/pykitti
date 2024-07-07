import os
import cv2
#Programm to add two pictures into one horizontally
def concatenate_images(folder1, folder2, output_folder):
    filenames1 = os.listdir(folder1)
    filenames2 = os.listdir(folder2)
    
    # Sortiere die Dateinamen, um sicherzustellen, dass sie paarweise zusammenpassen
    filenames1.sort()
    filenames2.sort()
    
    # Iteriere über die Dateinamen und füge die Bilder zusammen
    for filename1, filename2 in zip(filenames1, filenames2):
        # Pfade zu den Bildern
        image_path1 = os.path.join(folder1, filename1)
        image_path2 = os.path.join(folder2, filename2)
        
        # Lade die Bilder
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        
        # Überprüfe, ob die Bilder geladen wurden
        if image1 is None or image2 is None:
            print(f"Fehler beim Laden der Bilder: {image_path1}, {image_path2}")
            continue
        
        # Stelle sicher, dass die Bilder die gleiche Höhe haben
        min_height = min(image1.shape[0], image2.shape[0])
        image1 = image1[:min_height, :]
        image2 = image2[:min_height, :]
        
        # Verbinde die Bilder horizontal
        concatenated_image = cv2.hconcat([image1, image2])
        
        # Speichere das zusammengeführte Bild
        output_path = os.path.join(output_folder, f"{filename1}_{filename2}")
        cv2.imwrite(output_path, concatenated_image)
        
        print(f"Zusammengeführtes Bild gespeichert unter: {output_path}")


# Pfade zu den Bildern in den Ordnern
#rgb
folder1 = r"C:\Users\Besitzer\Desktop\pix2pix\image"#"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\image_00\data"
#lidar or depth
folder2 = r"C:\Users\Besitzer\Desktop\pix2pix\groundtruth_depth"#"D:\Dokumente\01_BA_Git\pykitti\ausgabe"

# Ausgabepfad für das kombinierte Bild
output_folder = r"C:\Users\Besitzer\Desktop\pix2pix\output"#"D:\Dokumente\01_BA_Git\pykitti\outputforpix2pix"
concatenate_images(folder1, folder2, output_folder)
