import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import cv2
from PIL import Image



# Funktion zum Laden der Transformationsmatrix (R|T) velo_to_cam aus einer Textdatei
def load_velo_to_cam(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        R_line = [float(x) for x in lines[1].strip().split()[1:]]
        T_line = [float(x) for x in lines[2].strip().split()[1:]]
        R = np.array(R_line).reshape(3, 3)
        T = np.array(T_line).reshape(3, 1)
    # Erstellen einer leeren 4x4-Matrix
        RT = np.eye(4)

        # Setzen der Rotationsmatrix in die oberen linken 3x3-Positionen
        RT[:3, :3] = R

        # Setzen des Translationsvektors in die vierte Spalte
        RT[:3, 3] = T.flatten()

    return RT

# Funktion zum Laden der Kalibrierungsparameter R_rect_00 aus einer Textdatei
def load_R_rect(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        R_line = [float(x) for x in lines[5].strip().split()[1:]]
        R_rect = np.array(R_line).reshape(3, 3)
        R_rect_00_padded = np.eye(4)
        R_rect_00_padded[:3, :3] = R_rect
        R_rect_00_padded[3, :3] = 0
        R_rect_00_padded[:3, 3] = 0
    return R_rect_00_padded

# Funktion zum Laden der Kalibrierungsparameter P_rect_xx aus einer Textdatei
def load_P_rect(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        P_line = [float(x) for x in lines[25].strip().split()[1:]]
        P_rect = np.array(P_line).reshape(3, 4)
    return P_rect

# Funktion zum Laden und Transformieren der Velodyne-Scandaten
def transform_velo_scan(scan_file, R_velo_to_cam, R_rect, P_rect):
    # Laden der Velodyne-Scandaten
    scan = np.fromfile(scan_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))  # Umformen in Punktwolke (x, y, z, reflectance)

    # Transformation der Punktwolke gemäß der gegebenen Transformation
    # Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X
    X = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))  # Erweitern der Punktwolke mit Einsen für die Homogenität
    Y = np.dot(P_rect, np.dot(R_rect, np.dot(R_velo_to_cam, X.T))).T

    return Y[:, :3]  # Rückgabe der transformierten Punktwolke (x, y, z)
'''
# Ordnerpfad mit Velodyne-Scandaten
folder_path = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"

# Dateipfade zu den Kalibrierungsdateien
file_path_R_velo_to_cam = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_velo_to_cam.txt"
file_path_R_rect =r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
file_path_P_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
output_folder = r'D:\Dokumente\01_BA_Git\pykitti\ausgabe'
'''
dataset = "2011_09_28_drive_0004_sync"
file_path_R_velo_to_cam = f"/Users/carstenschmotz/Desktop/kitti-step/val/{dataset}/calib_velo_to_cam.txt"
file_path_R_rect = f"/Users/carstenschmotz/Desktop/kitti-step/val/{dataset}/calib_cam_to_cam.txt"
file_path_P_rect = f"/Users/carstenschmotz/Desktop/kitti-step/val/{dataset}/calib_cam_to_cam.txt"
output_folder = f"/Users/carstenschmotz/Desktop/kitti-step/lidar_output/{dataset}"
folder_path =f"/Users/carstenschmotz/Desktop/kitti-step/val/{dataset}/velodyne_points/data"

def clear_and_create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
'''# Ordnerpfad mit Velodyne-Scandaten
folder_path = "/Users/carstenschmotz/Downloads/2011_09_26_drive_0009_sync-2/velodyne_points/data"

# Dateipfade zu den Kalibrierungsdateien
file_path_R_velo_to_cam = "/Users/carstenschmotz/Downloads/2011_09_26/calib_velo_to_cam.txt"
file_path_R_rect ="/Users/carstenschmotz/Downloads/2011_09_26/calib_cam_to_cam.txt"
file_path_P_rect = "/Users/carstenschmotz/Downloads/2011_09_26/calib_cam_to_cam.txt"
output_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/pix2pix_processing/output_folder_pointcloud"'''
clear_and_create_folder(output_folder)


# Laden der Kalibrierungsparameter
R_velo_to_cam = load_velo_to_cam(file_path_R_velo_to_cam)
R_rect = load_R_rect(file_path_R_rect)
P_rect = load_P_rect(file_path_P_rect)

  # Pfad zum Ausgabefolder

# Erstellen des Ausgabefolders, falls er nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Durchlaufen der Binärdateien im Ordner
for filename in os.listdir(folder_path):
    if filename.endswith(".bin"):
        file_path = os.path.join(folder_path, filename)
        
        transformed_scan = transform_velo_scan(file_path, R_velo_to_cam, R_rect, P_rect)
        valid_indices = transformed_scan[:, 2] > 0
        transformed_scan = transformed_scan[valid_indices]
        
        # Laden der Intensitätsdaten (vierte Spalte)
        intensity_data = np.fromfile(file_path, dtype=np.float32, count=-1)
        intensity_data = intensity_data.reshape(-1, 4)[:, 3]  # Vierte Spalte für Intensität
        intensity_data = intensity_data[valid_indices]
        
        # Konvertieren der Punktwolkenkoordinaten in Pixelkoordinaten
        pixel_coords = ((transformed_scan[:, :2] / transformed_scan[:, 2].reshape(-1, 1))).astype(int)
        
        # Erstellen eines leeren Bildes
        image_width = 1242
        image_height = 375  
        lidar_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)  # RGB-Bild
        
         # Färben der Punkte basierend auf Intensitäten
        # Färben der Punkte basierend auf Intensitäten
        for pixel_coord, intensity in zip(pixel_coords, intensity_data):
            x, y = pixel_coord
            
            if 0 <= x < image_width and 0 <= y < image_height:
                # Skalieren der Intensität auf den Bereich von 0 bis 255
                value = int(intensity * 255)
                if value <0 or value > 255:
                    raise ValueError("Value must be between 0 and 255")
                if value <= 127:
                    #Red to Green
                    t = value/ 127.0
                    color = np.array([255 * (1-t),255 * t,0])
                else:
                    #green to blue
                    t = (value -128)/127.0
                    color = np.array([0,255* (1-t), 255*t])
                #color.astype(int)
                
                #lidar_image[y, x] = color  # Setze den Pixelwert auf die entsprechende Farbe
                lidar_image[y, x] = color.astype(int)
                #255 0,0 blau ;0, 255 ,0 grün; 0,0,255 rot ; 0,255,255 gelb

            
            
        output_filename = os.path.join(output_folder, filename.replace('.bin', '.png'))
        cv2.imwrite(output_filename, lidar_image)
print("done")
        
        
        
        
        
        
       