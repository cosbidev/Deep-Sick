import shutil
import numpy as np
import PIL
from PIL import Image
import cv2
import argparse
from multiprocessing import Pool
import os
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 243603456
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


    img_array = np.array(PIL.Image.open(image_PATH), dtype=np.uint8)


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


def preprocess_openi(args):

    chunks_of_images, dimension, output_dir, skip_if_exists = args
    for img_path in tqdm(chunks_of_images):

        sub_dir_filename = img_path.split('images/')[-1]

        dest_img_path = os.path.join(output_dir, sub_dir_filename)

        # Process the image
        try:
            if skip_if_exists and os.path.exists(dest_img_path):
                #print(f"Skipping {img_path} as it already exists.")
                continue
            image = process_image(img_path, dimension)
            # Save the image
            # Get the parent directory of the image
            parent_dir = os.path.dirname(dest_img_path)

            # Save the image to png
            os.makedirs(parent_dir, exist_ok=True)
            image.save(dest_img_path, format='png', quality=95)

        except OSError as e:

            print(f"Error processing image: {e!r}", file=sys.stderr)
        except SyntaxError as e:
            print(f"Syntax error in image: {e!r}", file=sys.stderr)
        except PIL.UnidentifiedImageError as e:
            print(f"Unidentified image error in image: {e!r}", file=sys.stderr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Preprocess the Padchest dataset')
    parser.add_argument('--data_path', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/openi/images',
                        help='Path to the raw data of padchest')
    parser.add_argument('--output_path', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/openi',
                        help='Path to the output folder')
    parser.add_argument('--dimension', type=int, default=512,
                        help='Dimension of the output image')
    return parser.parse_args()




if __name__ == '__main__':
    # create_label_clinical_files('/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/openi/images-512')
    #
    args = parse_arguments()

    print('Started preprocessing of Openi')

    # Paths
    raw_data_path = args.data_path
    output_dir = args.output_path
    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'images-{args.dimension}'), exist_ok=True)
    path_to_images = os.path.join(output_dir, f'images-{args.dimension}')

    # Get all the files in the directories
    root_dir = Path(raw_data_path)
    images_list = [str(p) for p in root_dir.rglob("*") if p.suffix.lower() == ".png"]
    image_to_process = []
    for im in images_list: 
        image_name_extension = im.split('images/')[-1]
        final_im = os.path.join(path_to_images, image_name_extension)
        if not os.path.exists(final_im):
            image_to_process.append(im)

    # Create chunks
    num_processes = 6  #os.cpu_count() or 4  # Use all available cores or fallback
    chunk_size = max(1, len(image_to_process) // num_processes)
    image_list_chunks = [image_to_process[i:i + chunk_size] for i in range(0, len(image_to_process), chunk_size)]
    skip_if_exists = True
    # Wrap the chunks with args for the pool
    worker_args = [(chunk, args.dimension, path_to_images, skip_if_exists) for chunk in image_list_chunks]
    # Multiprocessing
    with Pool(processes=num_processes) as pool:
        pool.map(preprocess_openi, worker_args)
    print('Finished preprocessing of OpenI')
