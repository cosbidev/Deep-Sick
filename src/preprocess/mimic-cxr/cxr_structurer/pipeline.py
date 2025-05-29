import argparse
import os
import sys
import spacy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import walk_reports_mimic




mimic_dir_reports = "/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/mimic-cxr/mimic-cxr-reports/files"
#nlp('There is no focal consolidation, pleural effusion or pneumothorax.  Bilateral nodular opacities that most likely represent nipple shadows. The cardiomediastinal silhouette is normal.  Clips project over the left lung, potentially within the breast. The imaged upper abdomen is unremarkable. Chronic deformity of the posterior left sixth and seventh ribs are noted.')

parser = argparse.ArgumentParser(description="Open the MIMIC-CXR reports.")
parser.add_argument("--text_reports_dir", default=mimic_dir_reports, help="Directory containing report JSON files.")


if __name__ == "__main__":
    # Open all the reports in the MIMIC-CXR dataset
    args = parser.parse_args()
    mimic_dir_reports = args.text_reports_dir


    for pid, path, text in walk_reports_mimic(mimic_dir_reports):
        
        print(f"Patient ID: {pid}\nPath: {path}\nReport Sample: {text}")


    # Load NLP pipeline and linker once
    # nlp = spacy.load("en_ner_bionlp13cg_md")
    nlp = spacy.load("en_core_sci_sm")

    # This line takes a while, because we have to download ~1GB of data
    # and load a large JSON file (the knowledge base). Be patient!
    # Thankfully it should be faster after the first time you use it, because
    # the downloads are cached.
    # NOTE: The resolve_abbreviations parameter is optional, and requires that
    # the AbbreviationDetector pipe has already been added to the pipeline. Adding
    # the AbbreviationDetector pipe and setting resolve_abbreviations to True means
    # that linking will only be performed on the long form of abbreviations.
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    # Example usage
    text = ("There is no focal consolidation, pleural effusion or pneumothorax. "
            "Bilateral nodular opacities that most likely represent nipple shadows. "
            "The cardiomediastinal silhouette is normal. Clips project over the left lung, potentially within the breast. "
            "The imaged upper abdomen is unremark")

    # Process the text
    doc = nlp(text)

    # Print the entities and their UMLS concepts
    for entity in doc.ents:
        print(f"Entity: {entity.text}, Label: {entity.label_}")
        if entity._.kb_ents:
            print(f"UMLS Concepts: {entity._.kb_ents}")
