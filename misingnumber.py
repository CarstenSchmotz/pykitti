import os
import re

def find_missing_numbers(folder, start=0, end=794):
    # Regular expression to extract numbers from filenames
    number_pattern = re.compile(r'(\d+)')

    missing_files_report = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(folder):
        # Extract numbers from files
        file_numbers = []
        for file in files:
            match = number_pattern.search(file)
            if match:
                num = int(match.group(1))
                if start <= num <= end:
                    file_numbers.append(num)

        # Sort numbers and find missing ones in the range
        if file_numbers:
            file_numbers.sort()
            missing_numbers = [n for n in range(start, end + 1) if n not in file_numbers]
            if missing_numbers:
                relative_path = os.path.relpath(root, folder)
                missing_files_report.append((relative_path, missing_numbers))

    return missing_files_report

def print_report(missing_files_report):
    if missing_files_report:
        print("Missing files report:")
        for folder, missing_numbers in missing_files_report:
            print(f"In folder '{folder}': missing numbers {missing_numbers}")
    else:
        print("No missing files found.")

# Specify the folder to scan
folder_to_scan = r"D:\00_projekt_depth\data_depth_velodyne\train\2011_09_29_drive_0004_sync\proj_depth\velodyne_raw\image_02"

missing_files_report = find_missing_numbers(folder_to_scan)
print_report(missing_files_report)
