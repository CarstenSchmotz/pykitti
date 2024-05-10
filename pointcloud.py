import os
import numpy as np
import matplotlib.pyplot as plt

'''
def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def yield_velo_scans(velo_files):
    """Generator to parse velodyne binary files into arrays."""
    for file in velo_files:
        yield load_velo_scan(file)
        
        import os
import numpy as np
'''

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
        P_line = [float(x) for x in lines[9].strip().split()[1:]]
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

# Ordnerpfad mit Velodyne-Scandaten
folder_path = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data"

# Dateipfade zu den Kalibrierungsdateien
file_path_R_velo_to_cam = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_velo_to_cam.txt"
file_path_R_rect =r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"
file_path_P_rect = r"D:\Dokumente\01_BA_Git\pykitti\kitty\2011_09_26\2011_09_26_drive_0001_sync\calib_cam_to_cam.txt"

# Laden der Kalibrierungsparameter
R_velo_to_cam = load_velo_to_cam(file_path_R_velo_to_cam)
R_rect = load_R_rect(file_path_R_rect)
P_rect = load_P_rect(file_path_P_rect)

# Durchlaufen der Binärdateien im Ordner
for filename in os.listdir(folder_path):
    if filename.endswith(".bin"):
        file_path = os.path.join(folder_path, filename)
        
        # Transformation der Velodyne-Scandaten
        transformed_scan = transform_velo_scan(file_path, R_velo_to_cam, R_rect, P_rect)
        
        # Erstellen von Lidar-Bildern
        # Normalisieren der Punktwolke, um sie auf das Bild zu mappen (zum Beispiel auf das Intervall [0, 255])
        

        # Extrahiere x, y und z Koordinaten aus dem transformierten Scan
        x = transformed_scan[:, 0]
        y = transformed_scan[:, 1]
        z = transformed_scan[:, 2]
        '''
        # Erstelle ein 2D-Bild von x und y Koordinaten, wobei z als Farbkanal verwendet wird
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, c=z, cmap='viridis')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Lidar-Scan')
        plt.colorbar(label='Z-Koordinate')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        '''
        '''normalized_scan = transformed_scan - np.min(transformed_scan, axis=0)
        normalized_scan /= np.max(normalized_scan, axis=0)
        normalized_scan *= 255
        
        # Erstellen des Bildes
        lidar_image = normalized_scan[:, :3].astype(np.uint8)  # Nur die ersten drei Spalten (x, y, z) verwenden
        lidar_image = lidar_image.reshape((-1, 4))  # In ein Bildformat umformen
        
        # Anzeige oder Speicherung des Lidar-Bildes
        plt.imshow(lidar_image)
        plt.axis('off')
        plt.savefig(filename[:-4] + "_lidar_image.png", bbox_inches='tight', pad_inches=0)
        plt.show()'''
