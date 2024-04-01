import SimpleITK as sitk
import pandas as pd
import os
import csv
from scipy.stats import pearsonr

def generate_csv(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    with open('image_folder.csv', 'w', newline='') as csvfile:
        fieldnames = ['Patient', 'Image', 'Mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            subsubfolders = [f for f in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, f))]

            for subsubfolder in subsubfolders:
                subsubfolder_path = os.path.join(subfolder_path, subsubfolder)
            image_path = os.path.join(subfolder_path, 'orgimg.nrrd')
            mask_path = os.path.join(subfolder_path, 'segmentation.nrrd')

            writer.writerow({'Patient': subfolder, 'Image': image_path, 'Mask': mask_path})
            print(f'Added entry for {subfolder}')


def convert_labels(input_path, output_path):
    """
    Converts labels, according to different immunophenotyping.
    'D' and 'F' -> 0, others -> 1

    """
    df = pd.read_excel(input_path)
    for index, row in df.iterrows():
        if row['immunophenotyping'] in ['D', 'F']:
            df.at[index, 'label'] = 0
        else:
            df.at[index, 'label'] = 1
    df.to_excel(output_path, index=False)
    print(f"Processed file has been saved to: {output_path}")


# convert xlsx to csv
def xlsx_to_csv(input_path, output_path):  
    data_xlsx = pd.read_excel(input_path)
    data_xlsx.to_csv(output_path, index=False)
    print(f"csv file has been saved to: {output_path}")


# convert dicom to nrrd
def dicom2nrrd(csv_file_path, output_dir):
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        dicom_path = row["oripath"]
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            output_path = f"{output_dir}/image_{index}.nrrd"
            
            sitk.WriteImage(image, output_path)
            print(f":) Successfully converted to nrrd and saved to: {output_path}")
        except Exception as e:
            print(f":( Failed: {e}")


# Check if the dimensions of the image and mask match.
def check_dimension_match(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    mismatched_items = []
    for index, row in df.iterrows():
        image_path = row["oripath"]
        mask_path = row["maskpath"]
        
        try:
            image = sitk.ReadImage(image_path)
            mask = sitk.ReadImage(mask_path)
            
            if image.GetSize() != mask.GetSize():
                mismatched_items.append((image_path, mask_path))
        except Exception as e:
            print(f":( Failed to get: {e}")
            continue
    
    return mismatched_items


# Dilate and erode the masks to extract features 2mm adjacent to the tumor
def dilate_and_erode_masks(csv_file_path, resolution):
    data = pd.read_csv(csv_file_path)
    file_paths = data["mask"]
    
    dilation_voxels = int(1.0 / resolution)  # 1.0 for 2mm edge
    results = {}
    
    for file_path in file_paths:
        try:
            original_mask = sitk.ReadImage(file_path)

            dilation_radius = [dilation_voxels] * original_mask.GetDimension()

            dilated_mask = sitk.BinaryDilate(original_mask, dilation_radius)
            eroded_mask = sitk.BinaryErode(original_mask, dilation_radius)

            maskI2 = sitk.Subtract(dilated_mask, eroded_mask)

            output_file_pathI2 = file_path.replace("mask", "maskI2")
            sitk.WriteImage(maskI2, output_file_pathI2)
            results[file_path] = {
                "maskI2": output_file_pathI2
            }
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    return results


def calculate_pearson_coefficient(csv_file_path):
    df = pd.read_csv(csv_file_path)
    pearson_coefficients = {}
    for column in df.columns:
        if column != "VDT":
            coefficient, _ = pearsonr(df["VDT"], df[column])
            pearson_coefficients[column] = coefficient
    return pearson_coefficients




