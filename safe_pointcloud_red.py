import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Funktion zum Laden der Transformationsmatrix (R|T) velo_to_cam aus einer Textdatei
def load_velo_to_cam(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        R_line = [float(x) for x in lines[1].strip().split()[1:]]
        T_line = [float(x) for x in lines[2].strip().split()[1:]]
        R = np.array(R_line).reshape(3, 3)
        T = np.array(T_line).reshape(3, 1)
        RT = np.eye(4)
        RT[:3, :3] = R
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
        P_line = [float(x) for x in lines[9].strip().split()[1:]]
        P_rect = np.array(P_line).reshape(3, 4)
    return P_rect

# Funktion zum Laden und Transformieren der Velodyne-Scandaten
def transform_velo_scan(scan_file, R_velo_to_cam, R_rect, P_rect):
    scan = np.fromfile(scan_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))  
    X = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))  
    Y = np.dot(P_rect, np.dot(R_rect, np.dot(R_velo_to_cam, X.T))).T
    return Y[:, :3]  
os = 1 #windows = 1, mac=0
if os==1:
    folder_path = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"

    file_path_R_velo_to_cam = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_velo_to_cam.txt"
    file_path_R_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
    file_path_P_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
    output_folder = r'D:\Dokumente\01_BA_Git\pykitti\ausgabe'

else:
    folder_path = "/Users/carstenschmotz/Documents/GitHub/pykitti/kitty/2011_09_26/2011_09_26_drive_0001_sync/data"
    file_path_R_velo_to_cam = "/Users/carstenschmotz/Documents/GitHub/pykitti/kitty/2011_09_26/2011_09_26_drive_0001_sync/calib_velo_to_cam.txt"
    file_path_R_rect = "/Users/carstenschmotz/Documents/GitHub/pykitti/kitty/2011_09_26/2011_09_26_drive_0001_sync/calib_cam_to_cam.txt"
    file_path_P_rect = "/Users/carstenschmotz/Documents/GitHub/pykitti/kitty/2011_09_26/2011_09_26_drive_0001_sync/calib_cam_to_cam.txt"
    output_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/ausgabe"


R_velo_to_cam = load_velo_to_cam(file_path_R_velo_to_cam)
R_rect = load_R_rect(file_path_R_rect)
P_rect = load_P_rect(file_path_P_rect)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(folder_path):
    if filename.endswith(".bin"):
        file_path = os.path.join(folder_path, filename)
        
        transformed_scan = transform_velo_scan(file_path, R_velo_to_cam, R_rect, P_rect)
        valid_indices = transformed_scan[:, 2] > 0
        transformed_scan = transformed_scan[valid_indices]
        
        # Laden der Intensitätsdaten (vierte Spalte)
        intensity_data = np.fromfile(file_path, dtype=np.float32, count=-1)
        intensity_data = intensity_data.reshape(-1, 4)[:, 3]  # Vierte Spalte für Intensität
        
        # Konvertieren der Punktwolkenkoordinaten in Pixelkoordinaten
        scale_factor = 1
        pixel_coords = ((transformed_scan[:, :2] / transformed_scan[:, 2].reshape(-1, 1)) * scale_factor).astype(int)
        
        # Erstellen eines leeren Bildes
        image_width = 1242
        image_height = 375  
        lidar_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)  # RGB-Bild
        
         # Färben der Punkte basierend auf Intensitäten
        for pixel_coord, intensity in zip(pixel_coords, intensity_data):
            x, y = pixel_coord
            if 0 <= x < image_width and 0 <= y < image_height:
                # Skalieren der Intensität auf den Bereich von 0 bis 255
                intensity_scaled = int(intensity * 255)
                
                # Farbgebung basierend auf der Intensität
                if intensity_scaled == 0:
                    color = (0, 0, 0)  # Schwarz für fehlende Intensitäten
                elif intensity_scaled < 64:
                    color = (0, 0, intensity_scaled)  # Blau für niedrige Intensitäten
                elif intensity_scaled < 192:
                    color = (0, intensity_scaled, 255)  # Gelb für mittlere Intensitäten
                else:
                    color = (intensity_scaled, 255, 0)  # Rot für hohe Intensitäten
                
                lidar_image[y, x] = color  # Setze den Pixelwert auf die entsprechende Farbe
        output_filename = os.path.join(output_folder, filename.replace('.bin', '.png'))
        cv2.imwrite(output_filename, lidar_image)