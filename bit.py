import os
import cv2

def concatenate_images(image_paths, output_path):
    images = []
    
    # Laden der Bilder
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Fehler: Bild konnte nicht geladen werden: {image_path}")
            return
        images.append(image)

    # Überprüfen, ob die Bilder die gleiche Höhe haben
    image_heights = [image.shape[0] for image in images]
    if len(set(image_heights)) > 1:
        print("Fehler: Die Höhen der Bilder sind unterschiedlich.")
        return
    
    # Verbinden der Bilder nebeneinander
    concatenated_image = cv2.hconcat(images)

    # Speichern des kombinierten Bildes
    cv2.imwrite(output_path, concatenated_image)
    print(f"Das kombinierte Bild wurde erfolgreich unter {output_path} gespeichert.")

# Pfade zu den Bildern in den Ordnern
folder1_path = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\image_00\data"
folder2_path = r"D:\Dokumente\01_BA_Git\pykitti\ausgabe"

# Ausgabepfad für das kombinierte Bild
output_path = r"D:\Dokumente\01_BA_Git\pykitti\outputforpix2pix\concatenated_image.png"

# Sammeln der Bildpfade aus den Ordnern
image_paths_folder1 = [os.path.join(folder1_path, filename) for filename in os.listdir(folder1_path) if filename.endswith(".png")]
image_paths_folder2 = [os.path.join(folder2_path, filename) for filename in os.listdir(folder2_path) if filename.endswith(".png")]

# Kombinieren der Bildpfade aus beiden Ordnern
all_image_paths = image_paths_folder1 + image_paths_folder2

# Funktion aufrufen, um die Bilder zu kombinieren und das Ergebnis zu speichern
concatenate_images(all_image_paths, output_path)