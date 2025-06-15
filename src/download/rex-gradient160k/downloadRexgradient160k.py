from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
def read_nii_files(directory):
    """
    Retrieve paths of all NIfTI files in the given directory.

    Args:
    directory (str): Path to the directory containing NIfTI files.

    Returns:
    list: List of paths to NIfTI files.
    """
    nii_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, file))
    return nii_files



def download_images_files():

    repo_id = 'rajpurkarlab/ReXGradient-160K'

    hf_token = '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'
    ids_files = range(10)
    for i in tqdm(ids_files):

        filename = f'deid_png.part0{i}'
        if os.path.exists(f'/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/rex-gradient160k/{filename}'):
            print(f"File {filename} already exists, skipping download.")
            continue
        try:
            hf_hub_download(repo_id=repo_id,
                            repo_type='dataset',
                            token=hf_token,
                            filename=filename,
                            cache_dir='./',
                            local_dir='/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/rex-gradient160k'
                            )
        except Exception as e:
            print(f"Error downloading {e}")
            continue


if __name__ == "__main__":

    download_images_files()
