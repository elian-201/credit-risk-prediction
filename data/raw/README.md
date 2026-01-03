# Data Download Instructions

## Lending Club Dataset

The model uses the Lending Club loan dataset (2007-2018).

### Option 1: Download from Kaggle (Recommended)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/datasets/wordsforthewise/lending-club
3. Click "Download" to get the ZIP file
4. Extract `accepted_2007_to_2018Q4.csv` to this directory

### Option 2: Using Kaggle CLI

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle credentials
# Download kaggle.json from Kaggle (Account -> API -> Create New Token)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle datasets download -d wordsforthewise/lending-club
unzip lending-club.zip
```

### Option 3: Direct Download Link

If the above methods don't work, you can try:
- https://storage.googleapis.com/lending_club/LoanStats_2007_2018.csv

## Expected Files

After downloading, you should have:
- `accepted_2007_to_2018Q4.csv` (~2.3 GB, ~2.2 million rows)

## File Size Warning

The full dataset is approximately 2.3 GB. For testing:
- Use `nrows=10000` parameter when loading
- Or use `sample_frac=0.1` to load 10% of data

## Quick Test

```python
from src.data.ingestion import run_ingestion_pipeline

# Load small sample for testing
train_df, test_df = run_ingestion_pipeline(nrows=10000)
print(f"Train shape: {train_df.shape}")
print(f"Default rate: {train_df['target'].mean():.2%}")
```
