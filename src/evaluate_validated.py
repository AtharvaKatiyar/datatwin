"""
Validated evaluation module with enhanced metrics and fairness improvements.
Addresses scale sensitivity and sample size imbalance in distribution comparisons.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from scipy import stats
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_validated_metrics(real: pd.DataFrame, synthetic: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute validated metrics with normalized versions for fair comparison.
    
    Improvements over previous version:
    1. Adds normalized (scale-invariant) metrics alongside raw metrics
    2. Provides equal-sample-size KS distribution
    3. Separates physical impossibilities from observed range violations
    4. Documents all metric definitions precisely
    
    Args:
        real: Original real dataset
        synthetic: Generated synthetic dataset
        
    Returns:
        Dictionary with comprehensive validated evaluation metrics
    """
    logger.info("Computing validated evaluation metrics...")
    logger.info(f"Real data: {real.shape[0]} samples, Synthetic data: {synthetic.shape[0]} samples")
    
    metrics = {
        'sample_sizes': {
            'real': len(real),
            'synthetic': len(synthetic)
        },
        'per_column_metrics': {},
        'normalized_summary': {},
        'correlation_metrics': {},
        'quality_checks': {},
        'quality_column_analysis': {},
        'equal_sample_ks_distribution': {},
        'memorization': {}
    }
    
    # Standardize features for normalized metrics
    scaler = StandardScaler()
    real_normalized = pd.DataFrame(
        scaler.fit_transform(real),
        columns=real.columns
    )
    synthetic_normalized = pd.DataFrame(
        scaler.transform(synthetic),
        columns=synthetic.columns
    )
    
    # Per-column statistics (both raw and normalized)
    for col in real.columns:
        col_metrics = {}
        
        # Raw statistics
        col_metrics['raw'] = {
            'real_mean': float(real[col].mean()),
            'synthetic_mean': float(synthetic[col].mean()),
            'mean_diff': float(abs(real[col].mean() - synthetic[col].mean())),
            'real_std': float(real[col].std()),
            'synthetic_std': float(synthetic[col].std()),
            'std_diff': float(abs(real[col].std() - synthetic[col].std())),
            'real_median': float(real[col].median()),
            'synthetic_median': float(synthetic[col].median()),
            'median_diff': float(abs(real[col].median() - synthetic[col].median())),
            'real_min': float(real[col].min()),
            'synthetic_min': float(synthetic[col].min()),
            'real_max': float(real[col].max()),
            'synthetic_max': float(synthetic[col].max()),
        }
        
        # Normalized statistics (scale-invariant)
        col_metrics['normalized'] = {
            'mean_diff': float(abs(real_normalized[col].mean() - synthetic_normalized[col].mean())),
            'std_diff': float(abs(real_normalized[col].std() - synthetic_normalized[col].std())),
            'median_diff': float(abs(real_normalized[col].median() - synthetic_normalized[col].median())),
        }
        
        # Relative percentage differences
        if col_metrics['raw']['real_mean'] != 0:
            col_metrics['percent_mean_diff'] = float(
                abs(col_metrics['raw']['mean_diff'] / col_metrics['raw']['real_mean'] * 100)
            )
        else:
            col_metrics['percent_mean_diff'] = None
            
        if col_metrics['raw']['real_std'] != 0:
            col_metrics['percent_std_diff'] = float(
                abs(col_metrics['raw']['std_diff'] / col_metrics['raw']['real_std'] * 100)
            )
        else:
            col_metrics['percent_std_diff'] = None
        
        # KS test on full samples
        ks_stat, ks_p = stats.ks_2samp(real[col], synthetic[col])
        col_metrics['ks_statistic'] = float(ks_stat)
        col_metrics['ks_pvalue'] = float(ks_p)
        
        # Observed range violations (not necessarily invalid)
        below_min = int((synthetic[col] < real[col].min()).sum())
        above_max = int((synthetic[col] > real[col].max()).sum())
        col_metrics['observed_range_violations'] = {
            'below_min': below_min,
            'above_max': above_max,
            'total': below_min + above_max,
            'percentage': float((below_min + above_max) / len(synthetic) * 100)
        }
        
        # Physical constraint violations (feature-specific)
        physical_violations = 0
        if col in ['citric acid', 'volatile acidity', 'fixed acidity', 'chlorides', 
                   'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']:
            # These must be non-negative (concentrations/amounts)
            physical_violations = int((synthetic[col] < 0).sum())
        elif col == 'pH':
            # pH theoretically between 0-14
            physical_violations = int(((synthetic[col] < 0) | (synthetic[col] > 14)).sum())
        elif col == 'density':
            # Density around water, very unlikely outside [0.9, 1.1]
            physical_violations = int(((synthetic[col] < 0.9) | (synthetic[col] > 1.1)).sum())
        elif col == 'quality':
            # Quality must be in observed categories
            valid_qualities = set(real[col].unique())
            physical_violations = int((~synthetic[col].isin(valid_qualities)).sum())
        
        col_metrics['physical_constraint_violations'] = physical_violations
        
        # Unique value counts
        col_metrics['real_unique_count'] = int(real[col].nunique())
        col_metrics['synthetic_unique_count'] = int(synthetic[col].nunique())
        
        metrics['per_column_metrics'][col] = col_metrics
    
    # Equal-sample-size KS distribution (address sample size imbalance)
    logger.info("Computing equal-sample KS distribution (10 resamples)...")
    n_resamples = 10
    synthetic_size = len(synthetic)
    
    ks_distributions = {}
    for col in real.columns:
        ks_stats = []
        for _ in range(n_resamples):
            # Sample synthetic_size rows from real data
            real_sample = real[col].sample(n=synthetic_size, replace=False, random_state=np.random.randint(0, 10000))
            ks_stat, _ = stats.ks_2samp(real_sample, synthetic[col])
            ks_stats.append(ks_stat)
        
        ks_distributions[col] = {
            'mean': float(np.mean(ks_stats)),
            'std': float(np.std(ks_stats)),
            'min': float(np.min(ks_stats)),
            'max': float(np.max(ks_stats)),
            'samples': [float(x) for x in ks_stats],
            'description': f'{n_resamples} resamples of {synthetic_size} real rows vs {synthetic_size} synthetic rows'
        }
    
    metrics['equal_sample_ks_distribution'] = ks_distributions
    
    # Normalized summary statistics
    normalized_mean_diffs = [m['normalized']['mean_diff'] for m in metrics['per_column_metrics'].values()]
    normalized_std_diffs = [m['normalized']['std_diff'] for m in metrics['per_column_metrics'].values()]
    
    metrics['normalized_summary'] = {
        'avg_normalized_mean_diff': float(np.mean(normalized_mean_diffs)),
        'avg_normalized_std_diff': float(np.mean(normalized_std_diffs)),
        'definition': 'Computed on standardized features (mean=0, std=1) for scale-invariant comparison'
    }
    
    # Correlation analysis
    real_corr = real.corr()
    synthetic_corr = synthetic.corr()
    corr_diff_matrix = real_corr - synthetic_corr
    
    frobenius_norm = float(np.linalg.norm(corr_diff_matrix, 'fro'))
    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
    mean_abs_corr_error = float(np.abs(corr_diff_matrix.values[mask]).mean())
    max_abs_corr_error = float(np.abs(corr_diff_matrix.values[mask]).max())
    
    metrics['correlation_metrics'] = {
        'frobenius_norm': frobenius_norm,
        'mean_absolute_error': mean_abs_corr_error,
        'max_absolute_error': max_abs_corr_error,
        'formula_frobenius': 'sqrt(sum((C_real[i,j] - C_synthetic[i,j])^2 for all i,j))',
        'formula_mae': 'mean(|C_real[i,j] - C_synthetic[i,j]| for all i!=j)',
        'formula_max': 'max(|C_real[i,j] - C_synthetic[i,j]| for all i!=j)'
    }
    
    # Quality checks
    total_physical_violations = sum(m['physical_constraint_violations'] for m in metrics['per_column_metrics'].values())
    total_range_violations = sum(m['observed_range_violations']['total'] for m in metrics['per_column_metrics'].values())
    
    metrics['quality_checks'] = {
        'real_missing_values': int(real.isnull().sum().sum()),
        'synthetic_missing_values': int(synthetic.isnull().sum().sum()),
        'real_infinite_values': int(np.isinf(real.select_dtypes(include=[np.number])).sum().sum()),
        'synthetic_infinite_values': int(np.isinf(synthetic.select_dtypes(include=[np.number])).sum().sum()),
        'real_duplicate_rows': int(real.duplicated().sum()),
        'synthetic_duplicate_rows': int(synthetic.duplicated().sum()),
        'total_physical_constraint_violations': total_physical_violations,
        'total_observed_range_violations': total_range_violations,
        'note': 'Physical violations = impossible values (e.g. negative concentrations). Range violations = outside observed training range but may be physically valid.'
    }
    
    # Quality column analysis
    if 'quality' in real.columns:
        real_quality_dist = real['quality'].value_counts(normalize=True).sort_index()
        synthetic_quality_dist = synthetic['quality'].value_counts(normalize=True).sort_index()
        
        all_qualities = sorted(set(real_quality_dist.index) | set(synthetic_quality_dist.index))
        real_quality_dist = real_quality_dist.reindex(all_qualities, fill_value=0)
        synthetic_quality_dist = synthetic_quality_dist.reindex(all_qualities, fill_value=0)
        
        tvd = float(0.5 * np.abs(real_quality_dist - synthetic_quality_dist).sum())
        
        metrics['quality_column_analysis'] = {
            'real_distribution': {int(k): float(v) for k, v in real_quality_dist.items()},
            'synthetic_distribution': {int(k): float(v) for k, v in synthetic_quality_dist.items()},
            'total_variation_distance': tvd,
            'formula_tvd': '0.5 * sum(|P_real(x) - P_synthetic(x)| for all x)',
            'interpretation': 'TVD=0 means identical distributions, TVD=1 means completely disjoint'
        }
    
    # Memorization check
    logger.info("Checking memorization (nearest-neighbor distances)...")
    from scipy.spatial.distance import cdist
    
    real_norm = scaler.fit_transform(real)
    synth_norm = scaler.transform(synthetic)
    
    distances = cdist(synth_norm, real_norm, metric='euclidean')
    min_distances = distances.min(axis=1)
    
    threshold = 1e-6
    near_duplicates = (min_distances < threshold).sum()
    
    metrics['memorization'] = {
        'method': 'Nearest-neighbor Euclidean distance on standardized features',
        'distance_metric': 'Euclidean',
        'feature_normalization': 'StandardScaler (mean=0, std=1)',
        'threshold': threshold,
        'near_duplicate_count': int(near_duplicates),
        'near_duplicate_percentage': float(near_duplicates / len(synthetic) * 100),
        'min_distance_mean': float(min_distances.mean()),
        'min_distance_std': float(min_distances.std()),
        'min_distance_min': float(min_distances.min()),
        'min_distance_max': float(min_distances.max()),
        'note': 'This test does NOT prove privacy or rule out all memorization. It detects only exact or near-exact copies.'
    }
    
    logger.info(f"Physical constraint violations: {total_physical_violations}")
    logger.info(f"Observed range violations: {total_range_violations}")
    logger.info(f"Memorization near-duplicates: {near_duplicates}")
    
    return metrics


def create_per_feature_comparison_table(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Create detailed per-feature comparison table."""
    rows = []
    
    for col, col_metrics in metrics['per_column_metrics'].items():
        row = {
            'Feature': col,
            'Real_Mean': round(col_metrics['raw']['real_mean'], 4),
            'Synth_Mean': round(col_metrics['raw']['synthetic_mean'], 4),
            'Mean_Diff': round(col_metrics['raw']['mean_diff'], 4),
            'Percent_Mean_Diff': round(col_metrics['percent_mean_diff'], 2) if col_metrics['percent_mean_diff'] is not None else None,
            'Real_Std': round(col_metrics['raw']['real_std'], 4),
            'Synth_Std': round(col_metrics['raw']['synthetic_std'], 4),
            'Std_Diff': round(col_metrics['raw']['std_diff'], 4),
            'Percent_Std_Diff': round(col_metrics['percent_std_diff'], 2) if col_metrics['percent_std_diff'] is not None else None,
            'KS_Stat': round(col_metrics['ks_statistic'], 4),
            'KS_Pvalue': round(col_metrics['ks_pvalue'], 6),
            'KS_Mean_Equal_Sample': round(metrics['equal_sample_ks_distribution'][col]['mean'], 4),
            'Physical_Violations': col_metrics['physical_constraint_violations'],
            'Range_Violations': col_metrics['observed_range_violations']['total']
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def generate_validated_reports(
    metrics: Dict[str, Any],
    experiment_name: str,
    output_dir: str = "outputs"
):
    """Generate comprehensive validated reports."""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON report
    json_path = os.path.join(output_dir, f"evaluation_validated_{experiment_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"JSON report saved to {json_path}")
    
    # Per-feature CSV
    feature_table = create_per_feature_comparison_table(metrics)
    csv_path = os.path.join(output_dir, f"per_feature_comparison_{experiment_name}.csv")
    feature_table.to_csv(csv_path, index=False)
    logger.info(f"Per-feature table saved to {csv_path}")
    
    return json_path, csv_path


if __name__ == "__main__":
    from load_data import load_wine_dataset
    
    logger.info("Loading datasets...")
    real_data = load_wine_dataset()
    synthetic_data = pd.read_csv("outputs/synthetic/synthetic_improved_100.csv")
    
    logger.info("Computing validated metrics...")
    metrics = compute_validated_metrics(real_data, synthetic_data)
    
    logger.info("Generating reports...")
    generate_validated_reports(metrics, "improved_100", "outputs/reports_improved_100")
    
    logger.info("Validation complete!")
