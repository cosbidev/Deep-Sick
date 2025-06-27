import os
import subprocess
import argparse

def download_mimic_cxr_data(output_dir):
    try:
        output = subprocess.check_output([
            "gcloud", "storage", "--billing-project", "mimic-jpg-460612", "cp", "-r",
            "gs://brax-1.1.0.physionet.org/Anonymized_DICOMs/",
            "gs://brax-1.1.0.physionet.org/images/",
            "gs://brax-1.1.0.physionet.org/LICENSE.txt",
            "gs://brax-1.1.0.physionet.org/SHA256SUMS.txt",
            "gs://brax-1.1.0.physionet.org/master_spreadsheet_update.csv",
            output_dir
        ])
        print(output.decode())
    except subprocess.CalledProcessError as e:
        print("Error during initial download:\n", e.output.decode())

def check_and_download_missing_images(output_dir):
    image_file_path = os.path.join(output_dir, 'IMAGE_FILENAMES')
    not_found = 0

    if not os.path.exists(image_file_path):
        print(f"IMAGE_FILENAMES not found at {image_file_path}.")
        return

    with open(image_file_path) as f:
        for image in f.readlines():
            fname = image.strip()
            local_path = os.path.join(output_dir, fname)

            if not os.path.exists(local_path):
                not_found += 1
                print(f"{fname} not found. Downloading...")
                try:
                    output = subprocess.check_output([
                        "gcloud", "storage", "--billing-project", "mimic-jpg-460612", "cp",
                        f"gs://mimic-cxr-jpg-2.1.0.physionet.org/{fname}", local_path
                    ])
                    print(output.decode())
                except subprocess.CalledProcessError as e:
                    print(f"Failed to download {fname}:\n", e.output.decode())

    print(f"\nTotal missing files downloaded: {not_found}")

def main():
    parser = argparse.ArgumentParser(description="Download and verify MIMIC-CXR resources.")
    parser.add_argument('-o', '--output', required=True, help="Output directory to store files")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    download_mimic_cxr_data(output_dir)
    check_and_download_missing_images(output_dir)

if __name__ == "__main__":
    main()

