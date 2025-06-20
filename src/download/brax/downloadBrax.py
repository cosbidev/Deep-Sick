import argparse
import os
import subprocess
from tqdm import tqdm
# -------------------------------
# GCS Utilities
# -------------------------------

def list_gcs_dirs(gcs_path, billing_project):
    """List directories at a given GCS path."""
    try:
        result = subprocess.check_output([
            "gcloud", "storage", "ls", "--billing-project", billing_project, gcs_path
        ])
        lines = result.decode().splitlines()
        return [line.strip() for line in lines if line.endswith("/")]
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to list directories in {gcs_path}: {e}")
        return []

def download_gcs_resource(gcs_path, local_dir, billing_project):
    """Download a single GCS resource to a local directory."""
    try:
        subprocess.run([
            "gcloud", "storage", "cp", "-r", "--billing-project", billing_project,
            gcs_path, local_dir
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download from {gcs_path}: {e}")

# -------------------------------
# Main Download Function
# -------------------------------

def download_brax_data(output_dir, billing_project="mimic-jpg-460612"):
    os.makedirs(output_dir, exist_ok=True)
    print("Checking and downloading metadata and global files...")

    static_resources = {
        "images": "gs://brax-1.1.0.physionet.org/images",
        "LICENSE.txt": "gs://brax-1.1.0.physionet.org/LICENSE.txt",
        "SHA256SUMS.txt": "gs://brax-1.1.0.physionet.org/SHA256SUMS.txt",
        "master_spreadsheet_update.csv": "gs://brax-1.1.0.physionet.org/master_spreadsheet_update.csv"
    }

    for name, gcs_path in static_resources.items():
        local_path = os.path.join(output_dir, name)

        download_gcs_resource(gcs_path, output_dir, billing_project)

    """# Download image-level resources (images/ and Anonymized_DICOMs/)
    for img_dir in ["gs://brax-1.1.0.physionet.org/images/"]:  # , "gs://brax-1.1.0.physionet.org/Anonymized_DICOMs/"]:
        folder_name = os.path.basename(os.path.normpath(img_dir))
        local_img_dir = os.path.join(output_dir, folder_name)
        os.makedirs(local_img_dir, exist_ok=True)

        print(f"Listing directories in {img_dir}...")
        directories = list_gcs_dirs(img_dir, billing_project)

        print(f"Starting sequential download of {len(directories)} directories...")
        for dir_path in tqdm(directories) :
            patient_id = os.path.basename(os.path.normpath(dir_path))
            local_path = os.path.join(local_img_dir, patient_id)
            if os.path.exists(local_path):
                print(f"[SKIP] {patient_id} already exists.")
            else:
                #print(f"[DOWNLOAD] {patient_id} from {dir_path}")
                download_gcs_resource(dir_path, local_img_dir, billing_project)"""

# -------------------------------
# Entry Point
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and verify MIMIC-CXR resources.")
    parser.add_argument('-o', '--output', required=True, help="Output directory to store files")
    args = parser.parse_args()
    download_brax_data(args.output)
if __name__ == "__main__":
    main()
