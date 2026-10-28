"""
Data loading module for UCI Wine Quality dataset.
Handles downloading, caching, and inspection of the dataset.
"""

import os
import pandas as pd
import urllib.request
from typing import Dict, Any


UCI_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_DIR = "data"
CACHE_PATH = os.path.join(DATA_DIR, "winequality-red.csv")


def load_wine_dataset(force_download: bool = False) -> pd.DataFrame:
    """
    Load the UCI Wine Quality red wine dataset.
    
    Downloads from UCI repository if not cached locally, or if force_download is True.
    Handles semicolon-separated CSV format.
    
    Args:
        force_download: If True, download even if cached version exists
        
    Returns:
        DataFrame containing the wine quality dataset
        
    Raises:
        Exception: If download fails or file cannot be parsed
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Check if cached version exists and should be used
    if os.path.exists(CACHE_PATH) and not force_download:
        print(f"Loading cached dataset from {CACHE_PATH}")
        try:
            df = pd.read_csv(CACHE_PATH, sep=';')
            print(f"Successfully loaded {len(df)} records from cache")
            return df
        except Exception as e:
            print(f"Error loading cached file: {e}")
            print("Attempting to download fresh copy...")
    
    # Download the dataset
    print(f"Downloading dataset from {UCI_WINE_URL}")
    try:
        urllib.request.urlretrieve(UCI_WINE_URL, CACHE_PATH)
        print(f"Dataset downloaded and saved to {CACHE_PATH}")
    except Exception as e:
        error_msg = f"""
Failed to download dataset from {UCI_WINE_URL}
Error: {e}

Manual download instructions:
1. Visit https://archive.ics.uci.edu/dataset/186/wine+quality
2. Download the 'winequality-red.csv' file
3. Save it to: {os.path.abspath(CACHE_PATH)}
4. Run this script again
"""
        raise Exception(error_msg)
    
    # Load the downloaded dataset (semicolon-separated)
    try:
        df = pd.read_csv(CACHE_PATH, sep=';')
        print(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        raise Exception(f"Error parsing CSV file: {e}")


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect dataset and return comprehensive statistics.
    
    Args:
        df: DataFrame to inspect
        
    Returns:
        Dictionary containing:
        - shape: tuple of (rows, columns)
        - columns: list of column names
        - dtypes: dict of column name to data type
        - missing_values: dict of column name to count of missing values
        - duplicate_rows: count of duplicate rows
        - statistics: dict of descriptive statistics
    """
    inspection = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'statistics': df.describe().to_dict()
    }
    
    return inspection


def print_inspection_report(inspection: Dict[str, Any]) -> None:
    """
    Print a human-readable inspection report.
    
    Args:
        inspection: Dictionary returned by inspect_dataset()
    """
    print("\n" + "="*60)
    print("DATASET INSPECTION REPORT")
    print("="*60)
    
    print(f"\nShape: {inspection['shape'][0]} rows × {inspection['shape'][1]} columns")
    
    print(f"\nColumns ({len(inspection['columns'])}):")
    for col in inspection['columns']:
        dtype = inspection['dtypes'][col]
        missing = inspection['missing_values'][col]
        print(f"  - {col}: {dtype} ({missing} missing values)")
    
    print(f"\nDuplicate Rows: {inspection['duplicate_rows']}")
    
    print("\nBasic Statistics:")
    for col, stats in inspection['statistics'].items():
        print(f"\n  {col}:")
        for stat_name, value in stats.items():
            print(f"    {stat_name}: {value:.4f}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Load and inspect dataset
    df = load_wine_dataset()
    inspection = inspect_dataset(df)
    print_inspection_report(inspection)
