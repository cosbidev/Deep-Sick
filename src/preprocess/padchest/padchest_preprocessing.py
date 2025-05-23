
import os
import shutil

import pydicom as dicom
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2


def create_label_clinical_files(output_dir):
    """ Creates a label file from the meta data
    """
    # Output path
    print('Creating label file...')

    dest_path = os.path.join(os.path.dirname(output_dir), 'images')
    pass


def convert_dicom_to_image_AIforCovid(saved_patient: pd.DataFrame,
                                      cfg: dict,
                                      image_PATH: str
                                      ) -> (pd.DataFrame, Image.Image):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        path_to_data (str): Path to the dicom images
        extension (str): Extension of the image to save
        :param cfg: Configuration file
        :param saved_patient: DataFrame to save the patient information
    """
    extension = cfg['output']['extension']
    dcm = dicom.dcmread(image_PATH)
    try:
        img_array = dcm.pixel_array.astype(float)
        min_val, max_val = img_array.min(), img_array.max()

        interpretation = dcm.PhotometricInterpretation
        # Convert the image to a numpy array


        """# Crop Lung
        mask_path = os.path.join(cfg['output']['mask_dir'], image_PATH.split('/')[-1].replace('.dcm', '.tif'))
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(float)
        box_tot, _, _ = find_bboxes(mask)
        cropped_array = get_box(img=img_array, box_=box_tot, masked=False)"""
        cropped_array = img_array


        # Normalize the pixel values
        min_val = np.min(cropped_array)
        max_val = np.max(cropped_array)
        cropped_array -= min_val
        if max_val != min_val:  # Avoid division by zero
            cropped_array /= (max_val - min_val)
        cropped_array *= 255.0
        image_array = cropped_array.astype(np.uint8)


        # Invert images if necessary
        if interpretation == "MONOCHROME1":
            image_array = 255 - image_array


        # Histogram equalization
        image_array = cv2.equalizeHist(image_array)


        # Saving the new image
        dest_path = cfg['output']['img_dir']


        # Img original path
        original_path = image_PATH


        # Saving the new image
        patient_id_ext = image_PATH.split('/')[-1].replace('.dcm', extension)


        # Path file
        path_saving = os.path.join(dest_path, patient_id_ext)


        # Convert to JPG using OpenCV
        cv2.imwrite(path_saving, image_array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved_patient.loc[len(saved_patient)] = [patient_id_ext.split('.')[0], path_saving, original_path]

    except AttributeError as aerror:
        print(aerror)
        pass
    except RuntimeError as rerror:
        print(rerror)
        pass
    except InvalidDicomError as error:
        print(error)
        pass
    except ValueError as verror:
        print(verror)
        pass
    return saved_patient, Image.fromarray(image_array)


def preprocess_AIforCovid(**kwargs):
    """ Preprocesses the BrixIA dataset and saves it to disk
    """
    with open('configs/data/AFC_preprocessing.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    print('Started preprossesing of BrixIA')
    base_data_folder = '/Users/filruff/Documents/GitHub/COVID19-ItaChina/data/AIforCOVID'
    path_to_data_1 = os.path.join(base_data_folder, 'imgs')
    path_to_data_2 = os.path.join(base_data_folder, 'imgs_r2')
    path_to_data_3 = os.path.join(base_data_folder, 'imgs_r3')
    # CLINICAL DATA
    meta_path = os.path.join(base_data_folder, 'AIforCOVID.xlsx')
    meta_path_2 = os.path.join(base_data_folder, 'AIforCOVID_r2.xlsx')
    meta_path_3 = os.path.join(base_data_folder, 'AIforCOVID_r3.xlsx')

    # Clinical Data:
    clinical_meta_ = pd.read_excel(meta_path, engine='openpyxl')
    clinical_meta_2 = pd.read_excel(meta_path_2)
    clinical_meta_3 = pd.read_excel(meta_path_3)
    clinical_meta_global = pd.concat([clinical_meta_, clinical_meta_2, clinical_meta_3])
    reject_list = ['P_3_391', 'P_3_377', 'P_3_20', 'P_3_108', 'P_1_16', 'P_3_341', 'P_3_411', 'P_3_208', 'P_1_47', 'P_1_118' ]
    for pat in clinical_meta_global['ImageFile']:
        if pat in reject_list:
            clinical_meta_global = clinical_meta_global[clinical_meta_global['ImageFile'] != pat]
    os.makedirs(cfg['output']['clinical_dir'], exist_ok=True)
    clinical_meta_global.to_csv(os.path.join(cfg['output']['clinical_dir'], 'clinical_data_AFC.csv'))
    # Images:
    images_list_1 = [os.path.join(path_to_data_1, image_filename) for image_filename in os.listdir(path_to_data_1) if image_filename.replace('.dcm', '') not in reject_list]
    images_list_2 = [os.path.join(path_to_data_2, image_filename) for image_filename in os.listdir(path_to_data_2) if image_filename.replace('.dcm', '') not in reject_list]
    images_list_3 = [os.path.join(path_to_data_3, image_filename) for image_filename in os.listdir(path_to_data_3) if image_filename.replace('.dcm', '') not in reject_list]
    images_list = images_list_1 + images_list_2 + images_list_3
    images_list = sorted(images_list)
    saved_patient = pd.DataFrame(columns=['ID', 'Path', 'origin_Path'])

    dest_path = cfg['output']['img_dir']
    # CREATE FOLDER IF NOT EXIST
    if os.path.exists(dest_path) is True:
        shutil.rmtree(dest_path)
    os.makedirs(dest_path, exist_ok=True)


    anomaly_images = {center: [] for center in ['A', 'B', 'C', 'D', 'E', 'F']}
    cumulative_histograms = {center: np.zeros(256) for center in ['A', 'B', 'C', 'D', 'E', 'F']}
    for image in tqdm(images_list, desc='Converting .dcm to image {0}'.format(cfg['output']['extension'])):
        # Get the bounding box
        #box_tot = eval(bbox_pd[bbox_pd['img'] == image.split('/')[-1].replace('.dcm', '')]['all'].values[0])
        saved_patient, img_processed = convert_dicom_to_image_AIforCovid(saved_patient=saved_patient, image_PATH=image, cfg=cfg)

        histogram_image , _ = np.histogram(np.array(img_processed), bins=256, range=(0, 255))
        # Image center




        center_image_processed = clinical_meta_global[clinical_meta_global['ImageFile'] == image.split('/')[-1].replace('.dcm', '')]['Hospital'].values[0]

        if histogram_image.argmax() < 10:
            anomaly_images[center_image_processed].append(image.split('/')[-1].replace('.dcm', ''))
        # Append the histogram to the center
        cumulative_histograms[center_image_processed] += histogram_image

    # Aggregate the histograms
    for center, histograms in cumulative_histograms.items():
        normalized_histogram = histograms / histograms.sum()
        print(normalized_histogram.sum())
        plt.plot(normalized_histogram, label=center)
    plt.legend()
    plt.title('Plot Histograms Among 6 Centers')
    plt.show()

    # Label file creation


    print('Finished preprocessing.')


if __name__ == '__main__':
    preprocess_AIforCovid()
