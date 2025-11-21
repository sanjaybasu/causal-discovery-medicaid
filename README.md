# Automated Causal Discovery for Medicaid Population Health Programs

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Companion repository for**: "Automated Causal Discovery for Heterogeneous Treatment Effect Identification in Medicaid Population Health Programs: A Mechanistic Approach"

**Authors**: Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, Rajaie Batniji

**Submitted to**: American Journal of Epidemiology

---

## Overview

This repository contains code for applying automated causal discovery algorithms (Peter-Clark and Greedy Equivalence Search) to identify intervention-specific mechanisms explaining heterogeneous treatment effects in Medicaid population health programs.

### Key Findings

- **Therapy** reduces psychiatric admissions among older adults with recent hospitalizations (E-value 2.8)
- **Pharmacy** demonstrates dose-dependent cost reductions through medication adherence (E-value 3.1)
- **Community health workers** reduce ED visits addressing social determinants (E-value 3.4)
- **Care coordination** reduces ED visits among females through navigation support (E-value 2.6)

All mechanisms survived Benjamini-Hochberg false discovery rate correction and showed high bootstrap stability (82-98% discovery rates).

---

## Repository Structure

```
notebooks/causal_discovery/
├── __init__.py                      # Package initialization
├── data_loader_enhanced.py          # Data loading and temporal structuring
├── algorithms.py                    # PC and GES causal discovery algorithms
├── run_expanded_analysis.py         # Main analysis script
├── sensitivity_analyses.py          # Comprehensive sensitivity checks (propensity, falsification)
├── two_stage_mechanism_discovery.py # Two-stage HTE mechanism discovery
├── calculate_cates.py               # CATE calculation with bootstrap CIs
├── analyze_thresholds.py            # Empirical threshold analysis
├── test_algorithms.py               # Synthetic data validation
└── run_discovery.ipynb              # Interactive notebook

results/causal_discovery_expanded/
├── edge_lists/                      # Discovered edges (CSV)
├── mechanism_analysis/              # Intervention-specific pathways (JSON)
└── summary/                         # Analysis summaries (JSON)

figures/
├── figure1.png                      # Temporal tier structure
├── figure2.png                      # PC algorithm causal graph
└── figure3.png                      # GES algorithm causal graph
```

---

## Installation

### Requirements

- Python 3.10 or higher
- NumPy 1.24.3
- Pandas 2.0.2
- SciPy 1.10.1
- NetworkX 3.1
- Matplotlib 3.7.1
- Statsmodels 0.14.0

### Setup

```bash
# Clone repository
git clone https://github.com/waymarkcare/causal-discovery-medicaid.git
cd causal-discovery-medicaid

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start

```python
from notebooks.causal_discovery.data_loader_enhanced import load_and_prepare_data
from notebooks.causal_discovery.algorithms import PCAlgorithm, GESAlgorithm

# Load data with temporal structure
data, variable_info = load_and_prepare_data(
    data_dir='data/real_inputs',
    sample_size=5000
)

# Run PC algorithm
pc = PCAlgorithm(alpha=0.05, max_cond_size=3)
pc_graph = pc.learn_structure(
    data, 
    tier_structure=variable_info['tiers'],
    variable_names=variable_info['names']
)

# Run GES algorithm
ges = GESAlgorithm(score='bic', max_iter=100)
ges_graph = ges.learn_structure(
    data,
    tier_structure=variable_info['tiers']
)

# Analyze consensus mechanisms
from notebooks.causal_discovery.run_expanded_analysis import identify_consensus
consensus = identify_consensus(pc_graph, ges_graph)
```

### Running Full Analysis

```bash
cd notebooks/causal_discovery
python run_expanded_analysis.py --sample_size 5000 --output_dir ../../results/causal_discovery_expanded
```

### Running Sensitivity Analyses

To run the comprehensive suite of sensitivity checks (propensity score trimming, falsification tests, parameter sensitivity):

```bash
python sensitivity_analyses.py
```

### Running Two-Stage Mechanism Discovery

To execute the two-stage framework (HTE subgroup identification followed by subgroup-specific causal discovery):

```bash
python two_stage_mechanism_discovery.py
``````

**Arguments**:
- `--sample_size`: Number of members to sample (default: 5000)
- `--alpha`: Significance level for PC algorithm (default: 0.05)
- `--max_cond_size`: Maximum conditioning set size (default: 3)
- `--output_dir`: Directory for results (default: ../../results/causal_discovery_expanded)
- `--seed`: Random seed for reproducibility (default: 42)

### Output Files

The analysis generates:

1. **Causal Graphs**: PNG visualizations (figures/figure1.png, figure2.png, figure3.png)
2. **Edge Lists**: CSV files with discovered causal relationships
3. **Mechanism Analysis**: JSON files with intervention-specific pathways
4. **Summary Statistics**: JSON with algorithm convergence metrics

---

## Methodology

### Temporal Tier Structure

Variables are organized into three temporal tiers ensuring causal ordering:

- **Tier 0 (Baseline)**: Demographics, risk score, 6-month pre-activation utilization
- **Tier 1 (Treatment)**: Intervention exposure by specialty (therapy, pharmacy, CHW, care coordination)
- **Tier 2 (Outcomes)**: 6-month post-activation utilization

Temporal precedence constraints forbid edges from later to earlier tiers.

### Causal Discovery Algorithms

**Peter-Clark (PC) Algorithm**:
- Constraint-based approach using conditional independence tests
- Fisher Z-transformation of partial correlations
- Benjamini-Hochberg FDR correction (q=0.05)
- V-structure orientation and Meek rules

**Greedy Equivalence Search (GES)**:
- Score-based approach optimizing Bayesian Information Criterion
- Forward phase adds edges; backward phase removes
- Identical temporal constraints as PC

### Sensitivity Analyses

- **E-values**: Quantify robustness to unmeasured confounding
- **Bootstrap**: 1,000 iterations assess stability
- **Multiple testing**: Benjamini-Hochberg FDR at q=0.05
- **Varying α**: Test PC at 0.01, 0.05, 0.10

---

## Validation

### Synthetic Data Tests

Run validation on synthetic data with known causal structure:

```bash
python test_algorithms.py
```

This generates synthetic data from a predefined DAG and tests algorithm recovery accuracy.

**Expected Results**:
- Precision: ~85-90%
- Recall: ~75-85%
- F1-score: ~80-87%

### Real Data Analysis

Analyses use de-identified Medicaid claims data (not included due to privacy restrictions). Aggregate results and example outputs are provided in `results/`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{basu2025causal,
  title={Automated Causal Discovery for Heterogeneous Treatment Effect Identification in Medicaid Population Health Programs: A Mechanistic Approach},
  author={Basu, Sanjay and Patel, Sadiq Y and Sheth, Parth and Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and Batniji, Rajaie},
  journal={American Journal of Epidemiology},
  year={2025},
  note={Submitted}
}
```

---

## Data Availability

Individual-level Medicaid data cannot be shared due to patient privacy restrictions and data use agreements with state agencies. Researchers interested in replicating analyses may:

1. Apply for data access through state Medicaid agencies
2. Use the provided code with similar datasets
3. Contact the corresponding author for aggregate results

Synthetic example data is provided in `examples/` for code testing.

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Contact

**Corresponding Author**:  
Sanjay Basu, MD, PhD  
Waymark Care  
Email: sanjay.basu@waymarkcare.com

**Issues and Questions**:  
Please open an issue on GitHub or contact the corresponding author.

---

## Acknowledgments

We thank the Waymark Care clinical and data science teams for support in data collection and validation, and state Medicaid agencies for data access under data use agreements.

---

## Version History

- **v1.0.0** (2025-01): Initial release with AJE submission
  - PC and GES algorithms
  - Temporal constraint enforcement
  - Multiple testing correction
  - E-value sensitivity analysis
  - Bootstrap stability assessment

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Related Publications

1. Sheth P, Anders S, Basu S, Baum A, Patel SY. Comparing alternative approaches to care management prioritization: a prospective comparative cohort study of acute care utilization and equity among Medicaid beneficiaries. *Health Services Research*. In press.

2. Basu S, Patel SY, et al. Automated causal discovery for mechanistic insights in population health programs. *American Journal of Epidemiology*. Submitted 2025.
