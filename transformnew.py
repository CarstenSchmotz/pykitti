import pykitti
import numpy as np

#basedir = '/your/dataset/dir'
#basedir = 'Users/carstenschmotz/Desktop/kittydata/'
basedir = 'users/carstenschmotz/Dokumente/Github/kitty'
date = '2011_09_26'
drive = '0001_sync'

# The 'frames' argument is optional - default: None, which loads the whole dataset.
# Calibration, timestamps, and IMU data are read automatically. 
# Camera and velodyne data are available via properties that create generators
# when accessed, or through getter methods that provide random access.
data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))

# Transformation matrices
P_rect_xx = data.calib.P_rect_20
R_rect_00 = np.eye(4)  # Identity matrix for R_rect_00
R_velo_to_cam = data.calib.V2C
R_imu_to_velo = data.calib.R0_rect

# Function to perform the transformation
def transform_point(X, P_rect_xx, R_rect_00, R_velo_to_cam, R_imu_to_velo):
    # Transform point to camera coordinates
    Y_cam = np.dot(R_rect_00, np.dot(R_velo_to_cam, X))
    # Transform point to image plane coordinates
    Y_img = np.dot(P_rect_xx, Y_cam)
    # Normalize by dividing by the homogeneous coordinate
    Y_img /= Y_img[2]
    return Y_img[:2]

# Example point in velodyne coordinates [x, y, z, 1]
point_velo = np.array([0, 0, 0, 1])

# Transform point to camera image plane coordinates
point_image = transform_point(point_velo, P_rect_xx, R_rect_00, R_velo_to_cam, R_imu_to_velo)
print("Point in image plane coordinates:", point_image)

# Now you can use this transformed point to work with the image data
