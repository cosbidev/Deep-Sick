
import os
import shutil

import pydicom as dicom
import numpy as np
import pandas as pd
import yaml
import PIL
from PIL import Image
from pydicom.errors import InvalidDicomError
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import argparse
from multiprocessing import Pool
import os
import pandas as pd
from tqdm import tqdm
import sys

def create_label_clinical_files(output_dir):
    """ Creates a label file from the meta data
    """
    # Output path
    print('Creating label file...')

    dest_path = os.path.join(os.path.dirname(output_dir), 'images')
    pass


def process_image(
                    image_PATH: str,
                    output_dim: int,
                  ) -> (pd.DataFrame, Image.Image):
    """ Preprocesses a dicom image and saves it to disk as jpg

    Args:
        image_PATH (str): Path to the png image
        output_dim (int): Dimension of the output image
    """
    # Load the image


    img_array = np.array(PIL.Image.open(image_PATH), dtype=np.float32)

    # Normalize the pixel values
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    # Convert to float for safe arithmetic operations

    img_array -= min_val

    if max_val != min_val:  # Avoid division by zero
        img_array /= (max_val - min_val)

    # Scale to 0-255 range
    img_array *= 255.0
    # Scale to 8-bit range if needed, e.g., 0–65535
    image_array = img_array.astype(np.uint8)

    # Histogram equalization
    image_array = cv2.equalizeHist(image_array)

    # Pad the image to make it square
    h, w = image_array.shape[:2]  # Assuming grayscale; use [:2] for RGB too
    min_val = np.min(image_array)  # Or some predefined padding value

    if h > w:
        diff = h - w
        pad_left = diff // 2
        pad_right = diff - pad_left  # Handles odd case correctly
        image_array = np.pad(image_array, ((0, 0), (pad_left, pad_right)),
                             mode='constant', constant_values=min_val)

    elif w > h:
        diff = w - h
        pad_top = diff // 2
        pad_bottom = diff - pad_top  # Handles odd case correctly
        image_array = np.pad(image_array, ((pad_top, pad_bottom), (0, 0)),
                             mode='constant', constant_values=min_val)


    # Resize the image to n*n, with bilinear interpolation
    image_array =  cv2.resize(image_array, (output_dim, output_dim), interpolation=cv2.INTER_LINEAR)


    return Image.fromarray(image_array)


def preprocess_padchest(args):
    """ Preprocesses the BrixIA dataset and saves it to disk
    """
    chunks_of_images, dimension, output_dir, path_to_images  = args

    for img_path in tqdm(chunks_of_images):
        # Process the image
        try:
            image = process_image(img_path, dimension)
            # Save the image
            image_name = os.path.basename(img_path)
            # Save the image to png
            image.save(os.path.join(path_to_images, image_name), format='png')

        except OSError as e:

            print(f"Error processing image: {e!r}", file=sys.stderr)
        except SyntaxError as e:
            print(f"Syntax error in image: {e!r}", file=sys.stderr)
        except PIL.UnidentifiedImageError as e:
            print(f"Unidentified image error in image: {e!r}", file=sys.stderr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess the Padchest dataset')
    parser.add_argument('--data_path', type=str, default='/Volumes/Extreme SSD/Filippo/CXRs/PadChest',
                        help='Path to the raw data of padchest')
    parser.add_argument('--output_path', type=str, default='/Volumes/Extreme SSD/Filippo/CXRs/padchest_preprocessed',
                        help='Path to the output folder')
    parser.add_argument('--dimension', type=int, default=512,
                        help='Dimension of the output image')
    return parser.parse_args()

def main():
    args = parse_arguments()

    print('Started preprocessing of Padchest')

    # Paths
    raw_data_path = args.data_path
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'images-{args.dimension}'), exist_ok=True)
    path_to_images = os.path.join(output_dir, f'images-{args.dimension}')

    # Get all the zipped directories

    skip = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
            '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
            '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34',
            '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46']
    dirs_unzipped = [entry for entry in os.scandir(raw_data_path) if entry.path.endswith('.zip') and not entry.name.startswith('.')]

    # Filter out directories that start with '0' to '12'
    dirs_unzipped = [dir for dir in dirs_unzipped if not any([dir.name.split('.zip')[0] == s for s in skip])]

    for dir in dirs_unzipped:
        print(f'Processing directory: {dir.path}')

        # If the directory is not unzipped, unzip it
        unzipped_dir_path = dir.path[:-4]
        if not os.path.exists(unzipped_dir_path) and not os.path.isdir(unzipped_dir_path):
            # Unzip the directory
            print(f'Unzipping {dir.path}...')
            shutil.unpack_archive(dir.path, dir.path[:-4])
        else:
            print(f'Directory {dir.path[:-4]} already exists, skipping unzip.')
        # Now we have the unzipped directory
        unzipped_dir_path = dir.path[:-4]
        # Collect all image paths
        image_list_to_process = []
        print(f'Scanning directory: {unzipped_dir_path}')
        images_list = [entry.path for entry in os.scandir(unzipped_dir_path) if entry.is_file() and entry.name.endswith('.png') and not entry.name.startswith('.')]
        image_list_to_process.extend(images_list)

        # Create chunks
        num_processes = os.cpu_count() or 4  # Use all available cores or fallback
        chunk_size = max(1, len(image_list_to_process) // num_processes)
        image_list_chunks = [image_list_to_process[i:i + chunk_size] for i in range(0, len(image_list_to_process), chunk_size)]

        # Wrap the chunks with args for the pool
        worker_args = [(chunk, args.dimension, args.output_path, path_to_images) for chunk in image_list_chunks]

        # Multiprocessing
        with Pool(processes=num_processes) as pool:
            pool.map(preprocess_padchest, worker_args)

        # Remove the unzipped directory
        print(f'Removing unzipped directory: {dir.path}')
        shutil.rmtree(unzipped_dir_path, ignore_errors=True)

    print('Finished preprocessing of Padchest')

if __name__ == '__main__':
    main()