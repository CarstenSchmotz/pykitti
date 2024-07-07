import os
import cv2

def convert_to_8bit(folder_path):
    # Überprüfen, ob der Ordner existiert
    if not os.path.isdir(folder_path):
        print(f"Der angegebene Pfad {folder_path} ist kein Ordner.")
        return
    
    # Durchlaufen aller Dateien im Ordner
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Überprüfen, ob es sich um eine Bilddatei handelt
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Bild einlesen
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Fehler beim Lesen der Datei {file_path}. Möglicherweise kein unterstütztes Bildformat.")
                continue
            
            # Konvertierung in 8-Bit
            img_8bit = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))  # 16-Bit zu 8-Bit
            
            # Speichern des 8-Bit-Bildes im gleichen Ordner
            output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_8bit.png")
            cv2.imwrite(output_path, img_8bit)
            print(f"{filename} wurde erfolgreich zu 8-Bit konvertiert und als {os.path.basename(output_path)} gespeichert.")

# Beispielaufruf
folder_path = r"C:\Users\Besitzer\Desktop\pix2pix\output"#"D:\Dokumente\01_BA_Git\pykitti\outputforpix2pix"
output_folder_path = r"C:\Users\Besitzer\Desktop\pix2pix\output_bitrgb"
convert_to_8bit(folder_path)
