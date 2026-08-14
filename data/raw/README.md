# Raw data

This folder is where the raw dataset should be placed locally. It is not tracked in Git (see `.gitignore`) because the file is too large for GitHub.

## How to get it

1. Download the IBM AML HI-Medium dataset from Kaggle:
   https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

2. Place `HI-Medium_Trans.csv` (and `HI-Medium_Patterns.txt`, if used) in this folder.

3. Run `src/preprocessing/preprocessing_and_feature_engineering.ipynb` to generate the processed files.
