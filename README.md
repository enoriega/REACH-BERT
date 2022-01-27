# REACH BERT
Fine tune a BERT checkpoint on biochemical relation extractions 

## Scripts
- `fries_to_bert.py`: Transforms a fries output from `REACH` into CONLL format for sequence tagging. This will be the tranining data for the model.
- `mask_data_files.py`: Generates `N` copies of the unmasked data files following the masking parameters used by `RoBERTa`