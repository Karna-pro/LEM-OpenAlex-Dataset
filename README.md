# OpenAlex Data Collection Pipeline

This repository provides a lightweight pipeline for collecting and preprocessing scientific document data from the OpenAlex API.

## Overview

The script supports:

- Retrieving works from the OpenAlex API
- Reconstructing abstracts from `abstract_inverted_index`
- Extracting field (L1) and subfield (L2) labels
- Exporting the processed data into CSV format

## Data Source

The data is obtained from the OpenAlex API:  
https://api.openalex.org/

## Usage

```bash
python fetch_openalex_works.py \
    --email your_email@example.com \
    --target_size 1000 \
    --from_year 2018 \
    --output openalex_works.csv
```
## Notes

- OpenAlex metadata is continuously updated, and datasets constructed at different time points may exhibit slight differences.
