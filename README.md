# DataTwin - CTGAN Synthetic Data Generator

A validated Python prototype for generating synthetic tabular data using CTGAN (Conditional Tabular Generative Adversarial Network), with comprehensive evaluation metrics and academic-grade validation.

## 📋 Project Overview

DataTwin demonstrates the use of CTGAN for synthetic data generation on the UCI Wine Quality dataset. The project includes:

- ✅ Discrete column handling for categorical features
- ✅ Comprehensive evaluation metrics (marginal, correlation, quality distributions)
- ✅ Three validated training configurations (10, 100, 300 epochs)
- ✅ Complete reproducibility with fixed random seeds
- ✅ Academic-grade validation and honest limitation disclosure

## 🎯 Key Findings

**Mixed Performance Across Training Epochs:**
- **100 epochs**: Best marginal distribution matching (mean/std/median)
- **300 epochs**: Best correlation preservation (Frobenius norm)
- **10 epochs**: Best discrete distribution matching (quality TVD)

**No single model dominates all metrics** - selection depends on use case.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
cd datatwin

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Run single experiment (100 epochs recommended)
python run_experiments.py --experiments improved_100 --samples 500 --seed 42

# Run all three validated experiments
python run_experiments.py --experiments baseline_discrete,improved_100,improved_300 --samples 500 --seed 42

# Quick test with fewer epochs
python run_experiments.py --experiments baseline_discrete --samples 500 --seed 42
```

## 📊 Results Summary

### Verified Metrics (All Three Experiments)

| Metric | 10 Epochs | 100 Epochs | 300 Epochs | Winner |
|--------|-----------|------------|------------|--------|
| Avg Mean Diff | 2.18 | **0.75** | 2.24 | 100 |
| Avg Std Diff | 1.30 | **0.56** | 0.98 | 100 |
| Range Violations | 292 | **170** | 245 | 100 |
| Frobenius Norm | 3.17 | 3.26 | **3.01** | 300 |
| MAE Correlation | 0.21 | 0.22 | **0.20** | 300 |
| Quality TVD | **0.32** | 0.39 | 0.41 | 10 |
| Training Time | 0.29 min | 0.52 min | 1.56 min | 10 |

### Quality Distribution Matching

| Quality Score | Real % | 10-Epoch % | 100-Epoch % | 300-Epoch % |
|---------------|--------|------------|-------------|-------------|
| 3 | 0.63% | 9.60% | 10.60% | 8.40% |
| 4 | 3.31% | 13.40% | 12.80% | 21.20% |
| 5 | 42.59% | 24.80% | 23.80% | 21.40% |
| 6 | 39.90% | 25.60% | 20.00% | 20.40% |
| 7 | 12.45% | 18.20% | 19.80% | 17.00% |
| 8 | 1.13% | 8.40% | 13.00% | 11.60% |

**All models generate ONLY valid quality values [3, 8]** ✓

## 📁 Project Structure

```
datatwin/
├── README.md                          # This file
├── FINAL_VALIDATION_REPORT.md         # Complete validation report
├── VALIDATION_COMPLETE.md             # Validation summary
├── requirements.txt                   # Python dependencies
├── run_experiments.py                 # Main experiment runner
├── .env.example                       # Environment template
│
├── src/
│   ├── load_data.py                  # Dataset loading & inspection
│   ├── train_model_improved.py       # CTGAN training (discrete handling)
│   ├── generate_data.py              # Synthetic data generation
│   └── evaluate_validated.py         # Comprehensive evaluation metrics
│
├── data/
│   └── winequality-red.csv           # Cached UCI Wine Quality dataset
│
├── outputs/
│   ├── model/
│   │   ├── ctgan_model_baseline_discrete.pkl (10 epochs)
│   │   ├── ctgan_model_improved_100.pkl (100 epochs)
│   │   ├── ctgan_model_improved_300.pkl (300 epochs)
│   │   └── metadata_*.json
│   │
│   ├── synthetic/
│   │   ├── synthetic_baseline_discrete.csv
│   │   ├── synthetic_improved_100.csv
│   │   └── synthetic_improved_300.csv
│   │
│   ├── reports_{experiment}/
│   │   ├── evaluation_report.json
│   │   ├── evaluation_report.md
│   │   ├── evaluation_validated_{experiment}.json
│   │   └── per_feature_comparison_{experiment}.csv
│   │
│   ├── plots_{experiment}/           # 13 plots per experiment
│   ├── experiments_comparison.csv     # Summary comparison
│   └── EXPERIMENTS_SUMMARY.txt        # Quick reference
│
```

## 🔬 Evaluation Metrics

### Metric Definitions

**Marginal Distribution Metrics (Raw Units):**
- Mean Difference: `|mean(real) - mean(synthetic)|`
- Std Difference: `|std(real) - std(synthetic)|`
- Average computed across all 12 features

**Correlation Metrics:**
- Frobenius Norm: `sqrt(sum((C_real[i,j] - C_synthetic[i,j])^2))`
- Mean Absolute Error: `mean(|C_real[i,j] - C_synthetic[i,j]|)` for i≠j

**Quality Distribution:**
- Total Variation Distance: `0.5 * sum(|P_real(x) - P_synthetic(x)|)`
- Range: [0, 1] where 0=identical, 1=disjoint

**Data Quality:**
- Range violations: Values outside training data range
- Memorization: Nearest-neighbor distance on standardized features

### Validation Status

✅ All metrics independently verified from source CSVs  
✅ Quality distributions sum to 100%  
✅ TVD calculations confirmed (diff < 1e-6)  
✅ Configuration consistency verified across experiments  
✅ Zero near-duplicates detected (threshold: 1e-6 Euclidean distance)

## 🎓 Academic Use

### Citation

If using this code for academic work:

```
DataTwin CTGAN Prototype (2026)
Validated synthetic data generation using CTGAN v0.12.1
Dataset: UCI Wine Quality (red wine, 1,599 samples)
Experiments: 10/100/300 epochs with discrete column handling
```

### Validation Report

Complete methodology, results, and limitations documented in:
- `FINAL_VALIDATION_REPORT.md` - Comprehensive evaluation
- `VALIDATION_COMPLETE.md` - Verification summary
- `outputs/experiments_comparison.csv` - Metrics table

### Reproduction

All experiments fully reproducible with documented commands and fixed seeds.

## ⚠️ Limitations

### What This Project DOES Show
✓ CTGAN can generate synthetic tabular data with same schema  
✓ Discrete column handling prevents invalid categories  
✓ Training epochs affect different metrics differently  
✓ No obvious memorization (nearest-neighbor check)  
✓ Reproducible experiments with consistent evaluation

### What This Project DOES NOT Show
✗ Whether synthetic data is "good enough" for any use case  
✗ Privacy guarantees (no formal privacy analysis)  
✗ Downstream ML task performance  
✗ Optimal hyperparameters beyond epochs  
✗ Performance on other datasets  
✗ Comparison to other synthesis methods (TVAE, Copula, etc.)

### Known Issues

1. **Range Violations**: 170-292 values outside training range (not all physically impossible)
2. **Quality Distribution**: All models over-generate rare categories, under-generate common ones
3. **KS Tests**: All features show p<0.05 (statistically different distributions)
4. **Sample Size**: Only 500 synthetic samples generated (vs 1,599 real)
5. **Single Seed**: Results from seed=42 only (stability untested)

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Random seed for reproducibility
RANDOM_SEED=42

# Training configuration
BATCH_SIZE=500
EPOCHS=100  # Options: 10, 100, 300

# Generation configuration
N_SAMPLES=500

# Paths
DATA_DIR=data
OUTPUT_DIR=outputs
MODEL_DIR=outputs/model
```

### Hyperparameters

Default CTGAN parameters (can be modified in `src/train_model_improved.py`):
- Embedding dimension: 128
- Generator dimensions: (256, 256)
- Discriminator dimensions: (256, 256)
- Learning rate: 0.0002
- PAC: 10

## 📖 Usage Examples

### Load and Inspect Data

```python
from src.load_data import load_wine_dataset, inspect_dataset, print_inspection_report

df = load_wine_dataset()
inspection = inspect_dataset(df)
print_inspection_report(inspection)
```

### Train Custom Model

```python
from src.train_model_improved import train_ctgan_improved, save_model_with_metadata

model, metadata = train_ctgan_improved(
    df,
    epochs=100,
    discrete_columns=['quality'],
    experiment_name='custom',
    random_state=42
)

save_model_with_metadata(model, metadata, experiment_name='custom')
```

### Generate Synthetic Data

```python
from src.generate_data import load_model, generate_synthetic_data

model = load_model('outputs/model/ctgan_model_improved_100.pkl')
synthetic_data = generate_synthetic_data(model, n_samples=1000)
```

### Evaluate Quality

```python
from src.evaluate_validated import compute_validated_metrics, generate_validated_reports

metrics = compute_validated_metrics(real_data, synthetic_data)
generate_validated_reports(metrics, 'my_experiment', 'outputs/reports_my_experiment')
```

## 🤝 Contributing

This is an academic research prototype. For production use:

1. Implement post-processing (e.g., clip to valid ranges)
2. Tune hyperparameters for your specific dataset
3. Perform downstream task validation
4. Consider privacy analysis if required
5. Test multiple random seeds for stability

## 📚 References

- **CTGAN Paper**: Xu et al. (2019) - "Modeling Tabular data using Conditional GAN"
- **Dataset**: UCI Machine Learning Repository - Wine Quality Dataset
- **Library**: SDV CTGAN v0.12.1

## 📝 License

This project is for academic research and educational purposes.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- SDV project for the CTGAN implementation
- Kiro IDE for spec-driven development support

---

**Status**: ✅ Validated and Ready for Academic Review  
**Last Updated**: October 27, 2026  
**Version**: 1.0
