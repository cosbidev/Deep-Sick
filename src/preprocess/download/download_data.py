from datasets import load_dataset
import kagglehub
import os

os.getcwd()
# Path to data directory raw
raw_dir = "/Volumes/Extreme SSD/Filippo/Data/CXRs"






def download_chest_x_ray_data():

    # Download latest version

    print("Dataset already downloaded.")
    print("Downloading dataset...")
    path = kagglehub.dataset_download("nih-chest-xrays/data")

if __name__ == "__main__":
    # Chest X-ray 14 dataset
    download_chest_x_ray_data()



ds = load_dataset("valerieyuan/bimcv_covid19_all_cxr")
pass




