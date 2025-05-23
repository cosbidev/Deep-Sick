from multiprocessing import Pool
import os
import pandas as pd
from tqdm import tqdm


def worker(args):
    start_index, file = args
    #df_24h_literature = pd.read_csv(file)

    from google.cloud import storage

    project_id = 'mimic-jpg-460612'
    # Crea un client di storage
    storage_client = storage.Client(project_id)
    bucket_name = 'mimic-cxr-jpg-2.1.0.physionet.org'
    bucket = storage_client.bucket(bucket_name)
    os.system('gsutil -u ' + project_id + ' -m cp -r gs://mimic-cxr-2.0.0.physionet.org/files/p')


    download_directory = 'CXR/'
    for i in tqdm(range(start_index, start_index + 505)):
        subject_id = str(df_24h_literature.iloc[i]['subject_id'])
        # i primi tre caratteri di subject_id sono la cartella
        prefix = subject_id[:2]
        prefix_path = prefix
        study_id = str(df_24h_literature.iloc[i]['study_id'])
        # per ora tralascio un attimo il dicom_id, e scarico direttameete tutto lo studio
        # dicom_id = df_24h_literature.iloc[i]['dicom_id']
        temp_download_directory = download_directory + 'p' + prefix_path + '/p' + subject_id + '/s' + study_id + '/'
        os.makedirs(os.path.dirname(temp_download_directory), exist_ok=True)
        # eliminiamo da temp_download_directory tuttto quello che c'è dopo il penultimo '/'
        temp_download_directory = temp_download_directory[:temp_download_directory.rfind('/')]
        temp_download_directory = temp_download_directory[:temp_download_directory.rfind('/')]
        os.system('gsutil -u ' + project_id + ' -m cp -r gs://mimic-cxr-2.0.0.physionet.org/files/p')
        os.system('gsutil -u ' + project_id + ' -m cp -r gs://mimic-cxr-2.0.0.physionet.org/files/p' + prefix + '/p' + subject_id + '/s' + study_id + '.txt ' + temp_download_directory)


if __name__ == '__main__':
    worker((0, 'missing.csv'))
    files = ['modify.csv']
    arguments = []
    for i in range(0, len(files)):
        for j in range(0, 2020, 505):  # Assuming you want to process 4000 lines per file
            arguments.append((j, files[i]))
    with Pool(processes=1) as pool:  # Crea un pool di 40 worker
        pool.map(worker, arguments)

