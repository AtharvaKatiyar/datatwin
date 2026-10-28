"""
Improved model training module for CTGAN with discrete column handling.
Handles quality column as discrete and supports reproducible training.
"""

import os
import pandas as pd
from ctgan import CTGAN
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = "outputs/model"


def verify_ctgan_version():
    """Verify CTGAN library is installed and report version."""
    try:
        import ctgan
        version = ctgan.__version__
        logger.info(f"CTGAN version: {version}")
        return version
    except AttributeError:
        logger.warning("Could not determine CTGAN version")
        return "unknown"


def train_ctgan_improved(
    data: pd.DataFrame, 
    epochs: int = 300,
    discrete_columns: list = None,
    experiment_name: str = "default",
    batch_size: int = 500,
    random_state: int = 42
) -> tuple:
    """
    Train a CTGAN model with proper discrete column handling.
    
    Args:
        data: DataFrame containing training data
        epochs: Number of training epochs
        discrete_columns: List of column names to treat as discrete
        experiment_name: Name for this experiment (for saving)
        batch_size: Batch size for training
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (trained_model, training_metadata)
    """
    logger.info("="*80)
    logger.info(f"Training CTGAN - Experiment: {experiment_name}")
    logger.info("="*80)
    
    # Verify version
    version = verify_ctgan_version()
    
    # Set random seed for reproducibility
    import numpy as np
    import torch
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)
    
    logger.info(f"Random seed: {random_state}")
    logger.info(f"Dataset shape: {data.shape}")
    logger.info(f"Columns: {data.columns.tolist()}")
    
    if discrete_columns:
        logger.info(f"Discrete columns: {discrete_columns}")
        # Verify discrete columns exist
        for col in discrete_columns:
            if col not in data.columns:
                raise ValueError(f"Discrete column '{col}' not found in data")
            unique_values = sorted(data[col].unique())
            logger.info(f"  {col}: {len(unique_values)} unique values {unique_values}")
    else:
        logger.info("No discrete columns specified")
    
    # Initialize CTGAN
    logger.info(f"Initializing CTGAN with {epochs} epochs, batch_size={batch_size}...")
    model = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    logger.info(f"Training started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train the model with discrete columns
    if discrete_columns:
        model.fit(data, discrete_columns=discrete_columns)
    else:
        model.fit(data)
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now()
    training_duration = end_time - start_time
    
    logger.info(f"Training completed at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Training duration: {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
    
    # Collect metadata
    metadata = {
        'experiment_name': experiment_name,
        'ctgan_version': version,
        'epochs': epochs,
        'batch_size': batch_size,
        'random_state': random_state,
        'discrete_columns': discrete_columns,
        'training_samples': len(data),
        'num_features': len(data.columns),
        'features': data.columns.tolist(),
        'training_duration_seconds': training_duration,
        'training_duration_minutes': training_duration / 60,
        'start_time': start_datetime.isoformat(),
        'end_time': end_datetime.isoformat()
    }
    
    logger.info("Training metadata collected")
    
    return model, metadata


def save_model_with_metadata(
    model: CTGAN, 
    metadata: dict,
    experiment_name: str = "default"
) -> str:
    """
    Save trained CTGAN model and metadata to disk.
    
    Args:
        model: Trained CTGAN model instance
        metadata: Training metadata dictionary
        experiment_name: Name for this experiment
        
    Returns:
        Path to saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Model path
    model_path = os.path.join(MODEL_DIR, f"ctgan_model_{experiment_name}.pkl")
    
    logger.info(f"Saving model to {model_path}...")
    
    # Use CTGAN's built-in save method
    model.save(model_path)
    
    logger.info(f"Model saved successfully")
    
    # Verify file was created
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        logger.info(f"Model file size: {file_size / (1024*1024):.2f} MB")
    else:
        raise Exception(f"Model file was not created at {model_path}")
    
    # Save metadata as JSON
    import json
    metadata_path = os.path.join(MODEL_DIR, f"metadata_{experiment_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return model_path


if __name__ == "__main__":
    # Example usage with discrete column handling
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.load_data import load_wine_dataset
    
    logger.info("Loading dataset...")
    df = load_wine_dataset()
    
    # Train with quality as discrete column
    logger.info("Training CTGAN with quality as discrete column...")
    model, metadata = train_ctgan_improved(
        df, 
        epochs=10,
        discrete_columns=['quality'],
        experiment_name="test_discrete",
        random_state=42
    )
    
    logger.info("Saving model...")
    save_model_with_metadata(model, metadata, experiment_name="test_discrete")
    
    logger.info("Done!")
