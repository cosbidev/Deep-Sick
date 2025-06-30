# Instruction Tuning Data for CXR

## Data Summary

|                 | Total     |
|:----------------|:----------|
| train           | 4,263,566 |
| val             | 210,764   |
| test            | 177,542   |
| images (train)  | 908,103   |
| images (val)    | 48,539    |
| images (test)   | 66,654    |
| studies (train) | 1,097,080 |
| studies (val)   | 48,920    |
| studies (test)  | 62,454    |

## All the datasets

|                                                           | train     | val     | test    | images (train)   | images (val)   | images (test)   | studies (train)   | studies (val)   | studies (test)   |
|:----------------------------------------------------------|:----------|:--------|:--------|:-----------------|:---------------|:----------------|:------------------|:----------------|:-----------------|
| [Image Classification] [ChestXray14]                      | 43,345    | 11,211  | 22,425  | 43,345           | 11,211         | 22,425          | 43,345            | 11,211          | 22,425           |
| [Image Classification] [CheXpert-Public]                  | 182,802   | 196     | 468     | 216,034          | 225            | 612             | 182,802           | 196             | 468              |
| [Image Classification] [MIMIC-CXR]                        | 190,126   | 1,738   | 3,179   | 306,354          | 2,851          | 4,986           | 190,126           | 1,738           | 3,179            |
| [Image Classification] [PadChest]                         | 0         | 0       | 0       | 0                | 0              | 0               | 0                 | 0               | 0                |
| [Image Classification] [RSNA]                             | 18,678    | 4,003   | 4,003   | 18,678           | 4,003          | 4,003           | 18,678            | 4,003           | 4,003            |
| [Image Classification] [COVIDX-CXR-3]                     | 67,863    | 8,473   | 8,482   | 67,863           | 8,473          | 8,482           | 67,546            | 8,438           | 8,482            |
| [Image Classification] [CXR-LT]                           | 24,749    | 0       | 0       | 42,483           | 0              | 0               | 24,749            | 0               | 0                |
| [Image Classification] [Brax]                             | 5,364     | 0       | 1,399   | 10,728           | 0              | 2,798           | 5,364             | 0               | 1,399            |
| [Temporal Image Classification] [MS-CXR-T]                | 1,252     | 12      | 62      | 1,903            | 19             | 96              | 985               | 10              | 50               |
| [View Classification] [MIMIC-CXR]                         | 353,619   | 2,867   | 4,834   | 353,619          | 2,867          | 4,834           | 215,697           | 1,752           | 3,091            |
| [View Classification] [CheXpert-Public]                   | 223,397   | 234     | 234     | 223,397          | 234            | 234             | 223,397           | 234             | 234              |
| [Abnormality Detection] [VinDr-CXR]                       | 13,599    | 0       | 2,905   | 13,599           | 0              | 2,905           | 13,599            | 0               | 2,905            |
| [Abnormality Detection] [VinDr-PCXR]                      | 0         | 0       | 0       | 0                | 0              | 0               | 0                 | 0               | 0                |
| [Abnormality Grounding] [VinDr-CXR]                       | 30,282    | 0       | 4,022   | 11,657           | 0              | 1,988           | 4,510             | 0               | 937              |
| [Abnormality Grounding] [VinDr-PCXR]                      | 0         | 0       | 0       | 0                | 0              | 0               | 0                 | 0               | 0                |
| [Pneumothorax Segmentation] [SIIM]                        | 7,621     | 1,709   | 1,704   | 7,621            | 1,709          | 1,704           | 7,621             | 1,709           | 1,704            |
| [Foreign Object Detection] [Object-CXR]                   | 8,000     | 1,000   | 0       | 8,000            | 1,000          | 0               | 8,000             | 1,000           | 0                |
| [Phrase Grounding] [MS-CXR]                               | 964       | 7       | 189     | 878              | 5              | 164             | 878               | 5               | 164              |
| [Grounded Captioning] [MS-CXR]                            | 964       | 7       | 189     | 878              | 5              | 164             | 878               | 5               | 164              |
| [Grounded Diagnosis] [MS-CXR]                             | 964       | 7       | 189     | 878              | 5              | 164             | 878               | 5               | 164              |
| [Grounded Diagnosis] [VinDr-CXR]                          | 17,880    | 0       | 2,345   | 4,510            | 0              | 937             | 4,510             | 0               | 937              |
| [Grounded Diagnosis] [VinDr-PCXR]                         | 0         | 0       | 0       | 0                | 0              | 0               | 0                 | 0               | 0                |
| [Grounded Phrase Extraction] [MS-CXR]                     | 437       | 4       | 7       | 437              | 4              | 7               | 437               | 4               | 7                |
| [Findings Generation with Indication] [MIMIC-CXR]         | 97,790    | 755     | 1,277   | 173,331          | 1,339          | 2,114           | 97,790            | 755             | 1,277            |
| [Findings Generation] [MIMIC-CXR]                         | 214,832   | 1,745   | 3,086   | 354,523          | 2,874          | 4,879           | 214,832           | 1,745           | 3,086            |
| [Findings Generation] [CheXpert-Public]                   | 48,095    | 62      | 0       | 59,379           | 74             | 0               | 48,095            | 62              | 0                |
| [Findings Generation with Indication] [CheXpert-Public]   | 48,095    | 62      | 0       | 59,379           | 74             | 0               | 48,095            | 62              | 0                |
| [Findings Generation] [OpenI]                             | 0         | 0       | 3,337   | 0                | 0              | 6,473           | 0                 | 0               | 3,337            |
| [Findings Generation] [REX-Gradient]                      | 140,000   | 10,000  | 10,000  | 140,000          | 10,000         | 10,000          | 140,000           | 10,000          | 10,000           |
| [Impression Generation with Indication] [MIMIC-CXR]       | 85,380    | 671     | 957     | 155,305          | 1,215          | 1,665           | 85,380            | 671             | 957              |
| [Impression Generation] [MIMIC-CXR]                       | 156,117   | 1,258   | 1,976   | 274,237          | 2,180          | 3,362           | 156,117           | 1,258           | 1,976            |
| [Impression Generation] [CheXpert-Public]                 | 187,338   | 200     | 0       | 223,024          | 234            | 0               | 187,338           | 200             | 0                |
| [Impression Generation with Indication] [CheXpert-Public] | 187,338   | 200     | 0       | 223,024          | 234            | 0               | 187,338           | 200             | 0                |
| [Impression Generation] [OpenI]                           | 0         | 0       | 3,820   | 0                | 0              | 7,418           | 0                 | 0               | 3,820            |
| [Impression Generation] [REX-Gradient]                    | 140,000   | 10,000  | 10,000  | 140,000          | 10,000         | 10,000          | 140,000           | 10,000          | 10,000           |
| [Progression Findings Generation] [CheXpert-Public]       | 23,426    | 0       | 0       | 37,632           | 0              | 0               | 23,426            | 0               | 0                |
| [Progression Findings Generation] [MIMIC-CXR]             | 282,597   | 2,298   | 6,347   | 156,778          | 1,292          | 2,972           | 118,113           | 988             | 2,441            |
| [Progression Findings Generation] [REX-Gradient]          | 28,383    | 1,950   | 2,030   | 52,857           | 3,602          | 3,756           | 28,383            | 1,950           | 2,030            |
| [Progression Impression Generation] [MIMIC-CXR]           | 110,689   | 933     | 2,448   | 79,459           | 720            | 1,520           | 47,653            | 431             | 936              |
| [Progression Impression Generation] [CheXpert-Public]     | 108,858   | 0       | 0       | 144,000          | 0              | 0               | 108,858           | 0               | 0                |
| [Progression Impression Generation] [REX-Gradient]        | 20,126    | 1,342   | 1,474   | 37,863           | 2,500          | 2,754           | 20,126            | 1,342           | 1,474            |
| [Findings Summarization] [MIMIC-CXR]                      | 157,640   | 1,273   | 1,998   | 0                | 0              | 0               | 157,640           | 1,273           | 1,998            |
| [Findings Summarization] [OpenI]                          | 0         | 0       | 3,419   | 0                | 0              | 0               | 0                 | 0               | 27               |
| [Findings Summarization] [REX-Gradient]                   | 139,904   | 9,994   | 9,995   | 0                | 0              | 0               | 139,904           | 9,994           | 9,995            |
| [Open-Ended VQA] [Rad-Restruct]                           | 142,340   | 17,641  | 17,641  | 2,972            | 374            | 374             | 2,877             | 360             | 360              |
| [Open-Ended VQA] [MIMIC-CXR-VQA]                          | 251,569   | 61,076  | 11,028  | 125,784          | 8,592          | 500             | 125,784           | 8,592           | 500              |
| [Close-Ended VQA] [Rad-Restruct]                          | 142,340   | 17,641  | 17,641  | 2,972            | 374            | 374             | 2,877             | 360             | 360              |
| [Close-Ended VQA] [MIMIC-CXR-VQA]                         | 154,690   | 37,464  | 6,664   | 102,863          | 8,513          | 500             | 102,863           | 8,513           | 500              |
| [Difference VQA] [MIMIC-Diff-VQA]                         | 173,907   | 1,456   | 3,604   | 106,230          | 878            | 2,161           | 106,230           | 878             | 2,161            |
| [Text QA] [RadQA]                                         | 4,878     | 614     | 614     | 0                | 0              | 0               | 1,606             | 208             | 208              |
| [Report Evaluation] [ReXVal]                              | 0         | 0       | 200     | 0                | 0              | 0               | 0                 | 0               | 50               |
| [Natural Language Explanation] [MIMIC-NLE]                | 24,787    | 172     | 459     | 36,426           | 252            | 629             | 20,989            | 150             | 374              |
| [Natural Language Inference] [RadNLI]                     | 0         | 480     | 480     | 0                | 0              | 0               | 0                 | 480             | 480              |
| [Temporal Sentence Similarity] [MS-CXR-T]                 | 0         | 0       | 361     | 0                | 0              | 0               | 0                 | 0               | 361              |
| [Named Entity Recognition] [RadGraph]                     | 541       | 9       | 50      | 0                | 0              | 0               | 541               | 9               | 11               |
| Total                                                     | 4,263,566 | 210,764 | 177,542 | 908,103          | 48,539         | 66,654          | 1,097,080         | 48,920          | 62,454           |