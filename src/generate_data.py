"""
Synthetic data generation module.
Handles loading trained CTGAN models and generating synthetic samples.
"""

import os
import pandas as pd
from ctgan import CTGAN
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SYNTHETIC_DIR = "outputs/synthetic"
SYNTHETIC_PATH = os.path.join(SYNTHETIC_DIR, "synthetic_wine_data.csv")


def load_model(path: str) -> CTGAN:
    """
    Load a trained CTGAN model from disk.
    
    Args:
        path: File path to the saved model
        
    Returns:
        Loaded CTGAN model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model file is corrupted or cannot be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            f"Please train a model first using train_model.py"
        )
    
    logger.info(f"Loading model from {path}...")
    
    try:
        model = CTGAN.load(path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def generate_synthetic_data(model: CTGAN, n_samples: int) -> pd.DataFrame:
    """
    Generate synthetic data samples using the trained model.
    
    Args:
        model: Trained CTGAN model instance
        n_samples: Number of synthetic samples to generate
        
    Returns:
        DataFrame containing synthetic samples with same schema as training data
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    # Generate synthetic data using CTGAN's sample method
    synthetic_data = model.sample(n_samples)
    
    logger.info(f"Generated {len(synthetic_data)} samples")
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")
    logger.info(f"Columns: {synthetic_data.columns.tolist()}")
    
    # Verify we got the requested number of samples
    if len(synthetic_data) != n_samples:
        logger.warning(
            f"Expected {n_samples} samples but got {len(synthetic_data)}"
        )
    
    return synthetic_data


def save_synthetic_data(synthetic_data: pd.DataFrame, path: str = SYNTHETIC_PATH) -> None:
    """
    Save synthetic data to CSV file.
    
    Args:
        synthetic_data: DataFrame containing synthetic samples
        path: File path to save the CSV (default: outputs/synthetic/synthetic_wine_data.csv)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    logger.info(f"Saving synthetic data to {path}...")
    
    # Save to CSV
    synthetic_data.to_csv(path, index=False)
    
    logger.info(f"Synthetic data saved successfully")
    
    # Verify file was created
    if os.path.exists(path):
        file_size = os.path.getsize(path)
        logger.info(f"File size: {file_size / 1024:.2f} KB")
    else:
        raise Exception(f"Synthetic data file was not created at {path}")


def verify_schema_match(original: pd.DataFrame, synthetic: pd.DataFrame) -> bool:
    """
    Verify that synthetic data has the same schema as original data.
    
    Args:
        original: Original training data
        synthetic: Generated synthetic data
        
    Returns:
        True if schemas match, False otherwise
    """
    # Check column names
    if list(original.columns) != list(synthetic.columns):
        logger.error("Column names do not match!")
        logger.error(f"Original: {original.columns.tolist()}")
        logger.error(f"Synthetic: {synthetic.columns.tolist()}")
        return False
    
    # Check data types
    mismatched_types = []
    for col in original.columns:
        orig_dtype = original[col].dtype
        synth_dtype = synthetic[col].dtype
        if orig_dtype != synth_dtype:
            mismatched_types.append((col, orig_dtype, synth_dtype))
    
    if mismatched_types:
        logger.warning("Data type mismatches found:")
        for col, orig_dtype, synth_dtype in mismatched_types:
            logger.warning(f"  {col}: {orig_dtype} -> {synth_dtype}")
    
    logger.info("Schema verification passed: columns match")
    return True


if __name__ == "__main__":
    # Example usage
    from load_data import load_wine_dataset
    
    MODEL_PATH = "outputs/model/ctgan_model.pkl"
    
    logger.info("Loading trained model...")
    model = load_model(MODEL_PATH)
    
    logger.info("Generating synthetic data...")
    synthetic_data = generate_synthetic_data(model, n_samples=1000)
    
    logger.info("Verifying schema...")
    original_data = load_wine_dataset()
    verify_schema_match(original_data, synthetic_data)
    
    logger.info("Saving synthetic data...")
    save_synthetic_data(synthetic_data)
    
    logger.info("Done!")
