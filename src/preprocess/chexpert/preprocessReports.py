#%%
import json
import re
from nltk.tokenize import wordpunct_tokenize
import pandas as pd
import os
print(os.getcwd())
os.chdir('/Users/filruff/Desktop/PHD/PROGETTI/Deep-Sick/data/chexpert-public/chexpert-reports')
print(os.getcwd())
def format_chunks(tokens):
    # Step 1: Lowercase everything
    tokens = [t.lower() for t in tokens]

    # Step 2: Capitalize first word and each word after a period
    result = []
    capitalize_next = True
    for token in tokens:
        if capitalize_next and token.isalpha():
            result.append(token.capitalize())
            capitalize_next = False
        else:
            result.append(token)
        if token == '.':
            capitalize_next = True

    # Step 3: Join into a formatted string
    text = ' '.join(result)

    # Optional: clean spacing before punctuation
    text = text.replace(' .', '.').replace(' ,', ',').replace(' :', ':').replace(' ;', ';')

    return text

def radgraph_xl_preprocess_report(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\f", "  ")
    text = text.replace("\u2122", "      ")
    text = text.replace("\n", " ")
    text = text.replace("\\\"", "``").strip()

    text_sub = re.sub(r'\s+', ' ', text)
    text_sub = re.sub(r'\(.*?\)|\[.*?\]', '', text_sub)
    search = re.findall(r'(\b\d{1,2}\s*[-\/]\s*\d{1,2}(?:\s*[-\/]\s*\d{2,4})?\b|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:\s*,?\s*\d{2,4})?\b)', text_sub)

    # Slit the text based on '.'
    text_sub_chunks = re.split(r'[.,]+\s*', text_sub)

    # Remove empty chunks
    if len(text_sub_chunks) == 0:
        pass
    else:
        new_text_sub_chunks = []
        for chunk_sentence in text_sub_chunks:
            # Remove the phrase that contains the date
            if re.search(r'(\b\d{1,2}\s*[-\/]\s*\d{1,2}(?:\s*[-\/]\s*\d{2,4})?\b|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:\s*,?\s*\d{2,4})?\b)', chunk_sentence):
                # Remove the date from the sentence + separator
                fired_re = re.findall(r'(\b\d{1,2}\s*[-\/]\s*\d{1,2}(?:\s*[-\/]\s*\d{2,4})?\b|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?(?:\s*,?\s*\d{2,4})?\b)', chunk_sentence)
                for s in fired_re:
                    s = s.replace("/", r"\/")
                    old_sentence = chunk_sentence




                    #chunk_sentence = re.sub(rf'[\,\.\;]*\s*[\w*\s*\-*]*{s}*\s*[\s*\d*\,*\.*\;*\)]*?\s*[\w*\s\:*\-*]*', '', chunk_sentence)
                    chunk_sentence = re.sub("[\}\+\@*\(*\^\)\d\%\;\&\:\"\-\#\s\,\.\*\'\w*\/*]"+f"*\s*[\/\:\w*\s*\-*]*{s}\s*[\s*\d*\,*\.*\;\)\@]*\s*"+"[\w*\s*\-*\:*\/*\)\'\;\;\&\:\"\-\%\#\^\}]*", '', old_sentence)
                    # chunk_sentence = re.sub(f'[\,\.\;]*\s*[\w*\s*]*{s}[\w*\s*\:]*[,\.\;]*', '', chunk_sentence)
                    if '2(004)' in chunk_sentence:
                        chunk_sentence = ""
                    if len(chunk_sentence) != 0:
                        print(f"Old chunk sentence: {old_sentence}\n New chunk sentence: {chunk_sentence}")
                #new_sentence_white = re.sub(r'\A.*\d+\s*[-\/]\s*\d+\s*[-\/]?\s*\d*[\w*\s*]*[,\.]', '', chunk_sentence)
                new_text_sub_chunks.append(chunk_sentence)
            else:
                new_text_sub_chunks.append(chunk_sentence)
        text_sub = ' '.join(new_text_sub_chunks)
        # Remove anything inside parentheses (and ) or brackets [ ]
    # Remove the phrase that contains the date
    for s in search:
        text_sub = text_sub.replace(s, s.replace(" ", ""))
    # Remove extra spaces
    if len(search) > 0:
        for s in search:
            text_sub = text_sub.replace(s, s.replace(" ", ""))
    # Chunks
    tokenized = wordpunct_tokenize(text_sub)
    tokenized_text = format_chunks(tokenized)
    tokenized_text = tokenized_text.replace(").", ") .")
    tokenized_text = tokenized_text.replace("%.", "% .")
    tokenized_text = tokenized_text.replace(".'", ". '")
    tokenized_text = tokenized_text.replace("%,", "% ,")
    tokenized_text = tokenized_text.replace("%)", "% )")
    return tokenized_text

    # %%

    # Load the CSV file into a DataFrame
    df = pd.read_csv('df_chexpert_plus_240401.csv')

    # --- Preprocess the findings text
    df_findings = df[df['section_findings'].apply(lambda x: isinstance(x, str) and len(x.split()) >= 2)]

    # Preprocess the 'section_findings' and 'section_impression' columns
    df_findings.loc[:,'section_findings'] = df_findings['section_findings'].apply(lambda x: radgraph_xl_preprocess_report(x))
    df_findings['section_impression'] = df_findings['section_impression'].apply(lambda x: radgraph_xl_preprocess_report(x))
    df_findings.reset_index(drop=True, inplace=True)
#%%
    #df_impression.loc['section_impression'] = df_impression['section_impression'].apply(lambda x: radgraph_xl_preprocess_report(x))
    print(f"Number of findings: {len(df_findings)}")

    # Load radgraph-XL annotations
    with open("radgraph-XL-annotations/section_findings.json") as f:
        findings = json.load(f)
    with open("radgraph-XL-annotations/section_impression.json") as f:
        impressions = json.load(f)

    df_findings['section_findings_radgraph'] = df_findings['section_findings'].astype(str)
    for ix, (find, impress) in enumerate(zip(findings, impressions)):
        finding = find['0']['text']
        impression = impress['0']['text']
        df_findings.loc[ix, 'section_findings_radgraph'] = finding
        df_findings.loc[ix, 'section_impression_radgraph'] = impression
        # Add the path to the image and other metadata to the annotations file
        findings[ix]['0']['path_to_image'] = df_findings.loc[ix, 'path_to_image']
        impressions[ix]['0']['path_to_image'] = df_findings.loc[ix, 'path_to_image']



    # Save to a new CSV file
    df_findings.to_csv('df_chexpert_plus_240401_preprocessed.csv', index=False)
    # Save a new JSON file with the annotations
    with open('radgraph-XL-annotations/section_findings_preprocessed.json', 'w') as f:
        json.dump(findings, f, indent=4)
    with open('radgraph-XL-annotations/section_impression_preprocessed.json', 'w') as f:
        json.dump(impressions, f, indent=4)

#%%
