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
        P_line = [float(x) for x in lines[p_rec_choice].strip().split()[1:]]
        P_rect = np.array(P_line).reshape(3, 4)
    return P_rect

# Funktion zum Laden und Transformieren der Velodyne-Scandaten
def transform_velo_scan(scan_file, R_velo_to_cam, R_rect, P_rect):
    scan = np.fromfile(scan_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))  
    X = np.hstack((scan[:, :3], np.ones((scan.shape[0], 1))))  
    Y = np.dot(P_rect, np.dot(R_rect, np.dot(R_velo_to_cam, X.T))).T
    return Y[:, :3]  
system_choice = 0 #windows = 1, mac=0
p_rec_choice = 33 #9 00 33 for 03
if system_choice==1:
    folder_path = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"

    file_path_R_velo_to_cam = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_velo_to_cam.txt"
    file_path_R_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
    file_path_P_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
    output_folder = r'D:\Dokumente\01_BA_Git\pykitti\ausgabe'

else:
    folder_path = "/Users/carstenschmotz/Downloads/2011_09_26-2/2011_09_26_drive_0001_sync/velodyne_points/data"
    file_path_R_velo_to_cam = "/Users/carstenschmotz/Downloads/2011_09_26/calib_velo_to_cam.txt"
    file_path_R_rect = "/Users/carstenschmotz/Downloads/2011_09_26/calib_cam_to_cam.txt"
    file_path_P_rect = "//Users/carstenschmotz/Downloads/2011_09_26/calib_cam_to_cam.txt"
    output_folder = "/Users/carstenschmotz/Documents/GitHub/pykitti/ausgabe_try"

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
        