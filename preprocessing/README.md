## Pre-processing codes for SeqTagQA (Yoon et al. 2022) 

Source codes and resources for [Sequence Tagging for Biomedical Extractive Question Answering.](https://arxiv.org/abs/2104.07535)

Please cite:
 ```bib
 @article{yoon-etal-2022-sequence,
    author = {Yoon, Wonjin and Jackson, Richard and Lagerberg, Aron and Kang, Jaewoo},
    title = "{Sequence tagging for biomedical extractive question answering}",
    journal = {Bioinformatics},
    year = {2022},
    month = {06},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac397},
}
 ```

#### Please note that this pre-processing code is a pre-release version.

### Resource file
You will need a resource file (we recommend naming it "pubmedDict.json") containing the documents from the BioASQ training and test dataset.

Please download it using the following command:
```bash
wget https://wonjin.info/bioasq/pubmedDict.json
```
<br>LICENSE Note: Downloading pubmedDict.json indicates your acceptance of the <a href="https://www.nlm.nih.gov/databases/download/terms_and_conditions.html">Terms and Conditions</a> from National Library of Medicine.

**To be announced soon:** We will release the code and instructions for creating pubmedDict.json, which involves collecting documents using the Entrez library.

### Scripts

1. First, convert the BioASQ format dataset to "qca-triplets". (Note that "qca-triplets" format is not the same as SQuAD format, but it resembles SQuAD format.)

Here is an example CLI script:
```bash
python convert_bioasq-list_to_qca-triplet.py \
 --input_path ../../BioASQ-original/10B1_golden.json \
 --resource_path ../../resources/pubmedDict.json \
 --output_path outputs/qca-test.json \
 --test
```
**IMPORTNAT:** Use the `--test` flag for test files. If you are generating a file for the challenge (where the answers are not included in the file), use the `--test_without_answer` flag. 

**To be announced soon: conversion code from SQuAD to "qca-triplets".**

2. Next, convert the qca-triplets (`outputs/qca-test.json`) to CoNLL-QA format, which is the format used for the SeqTagQA train/test code.

Here is an example CLI script:
```bash
python convert_qca-triplet_to_CoNLL-QA.py \
 --input_path=outputs/cqa-test.json \
 --output_path=outputs/test.tsv
```
In this step, you do not need to worry about flags for test files.

