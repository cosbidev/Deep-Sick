import os



def open_report_mimim(report_path):
    """Open a report file and return its content."""
    with open(report_path, 'r') as file:
        content = file.read()
    return content

def clean_text_findings_impression(text):
    """Clean the text by removing unnecessary characters and formatting."""
    # Remove newlines and extra spaces
    cleaned_text = ' '.join(text.split())
    # Remove any leading or trailing whitespace
    cleaned_text = cleaned_text.strip()
    return cleaned_text


def all_the_reports_for_patient(patient_common_code_dirs):
    """Get all report files for a given patient ID."""

    for dir_name_path in patient_common_code_dirs:
        # Patient Mimic CXR reports directory
        patients_scanned = os.scandir(dir_name_path.path)

        



        if os.path.exists(mimic_dir_reports):
            return mimic_dir_reports


    patient_reports = []
    for root, dirs, files in os.walk(mimic_dir_reports):
        for file in files:
            if file.startswith(f'{patient_id}-'):
                patient_reports.append(os.path.join(root, file))
    return patient_reports