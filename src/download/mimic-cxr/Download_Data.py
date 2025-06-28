import os
import pandas as pd
from tqdm import tqdm

df_24h_literature = pd.read_csv('MIMIC Demo/cxr_metadata_merged_vitalsign_cleaned_24h_literature.csv')

# per ogni riga di df_24h_literature, controlliamo se i file sono presenti sull'hard disk, quelli che non ci sono li aggiungimao a un altro dataframe
# creiamo un dataframe vuoto
df_missing = pd.DataFrame(columns=['subject_id', 'study_id', 'dicom_id'])
for i in tqdm(range(len(df_24h_literature))):
    subject_id = str(df_24h_literature.iloc[i]['subject_id'])
    prefix = subject_id[:2]
    if prefix == '16':
        prefix = '16-3'
    study_id = str(df_24h_literature.iloc[i]['study_id'])
    dicom_id = str(df_24h_literature.iloc[i]['dicom_id'])
    temp_download_directory = 'E:/CXR/physionet.org/files/mimic-cxr/2.0.0/files/p' + prefix + '/p' + subject_id + '/s' + study_id + '/' + dicom_id + '.dcm'
    if not os.path.exists(temp_download_directory):
        df_missing = df_missing._append({'subject_id': subject_id, 'study_id': study_id, 'dicom_id': dicom_id}, ignore_index=True)

df_missing.to_csv('MIMIC Demo/missing.csv', index=False)

# dividiamo df_missing in 10 parti uguali
df_24h_literature = pd.read_csv('MIMIC Demo/missing.csv')

# dividiamo df_missing in 10 parti uguali che chiamaeremo 10.csv, 11.csv, 12.csv, 13.csv, 14.csv, 15.csv, 16.csv, 17.csv, 18.csv, 19.csv
for i in range(10):
    df_missing.iloc[i*len(df_missing)//10:(i+1)*len(df_missing)//10].to_csv('MIMIC Demo/' + str(i+10) + '.csv', index=False)

# estraiamo tutti quelli che iniziano con 17
df_24h_literature = pd.read_csv('MIMIC Demo/missing.csv')
# trasformiamo subject_id in stringa
df_missing['subject_id'] = df_missing['subject_id'].astype(str)
missing = df_missing[df_missing['subject_id'].str.startswith('16')]
missing.to_csv('MIMIC Demo/16.csv', index=False)
