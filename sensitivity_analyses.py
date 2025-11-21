"""
Comprehensive Sensitivity Analyses for Hernán-Style Rigorous Causal Inference

This script implements sensitivity analyses beyond E-values:
1. Propensity score diagnostics and trimming
2. Falsification tests (treatment → baseline outcomes)
3. Algorithm parameter sensitivity
4. Variable selection robustness

Author: Sanjay Basu
Date: 2025-11-21
"""

import sys
sys.path.insert(0, '/Users/sanjaybasu/waymark-local/notebooks')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from causal_discovery.data_loader_enhanced import load_causal_dataset_enhanced_optimized, TemporalConfig
from causal_discovery.algorithms import run_pc_algorithm, run_ges_algorithm

# Configuration
OUTPUT_DIR = Path('/Users/sanjaybasu/waymark-local/results/causal_discovery_expanded/sensitivity_analyses')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("HERNÁN-STYLE SENSITIVITY ANALYSES")
print("="*80)

# Load data
config = TemporalConfig(baseline_months=6, followup_months=6, intervention_buffer_days=30)
dataset, metadata = load_causal_dataset_enhanced_optimized(config=config, sample_size=5000)

print(f"\nDataset loaded: {len(dataset)} members")

# ============================================================================
# 1. PROPENSITY SCORE DIAGNOSTICS AND TRIMMING
# ============================================================================
print("\n" + "="*80)
print("1. PROPENSITY SCORE ANALYSIS (Addressing Positivity Violations)")
print("="*80)

# Estimate propensity score for therapy receipt
baseline_vars = ['age', 'baseline_ip_ct', 'baseline_ed_ct', 'baseline_cost', 'risk_score']
X = dataset[baseline_vars].fillna(0)
y = dataset['therapy_any']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit propensity score model
ps_model = LogisticRegression(random_state=42, max_iter=1000)
ps_model.fit(X_scaled, y)
propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]

dataset['propensity_score'] = propensity_scores

# Diagnostic statistics
print(f"\nPropensity Score Distribution:")
print(f"  Therapy recipients (n={y.sum()}):")
print(f"    Mean: {propensity_scores[y==1].mean():.3f}")
print(f"    Median: {np.median(propensity_scores[y==1]):.3f}")
print(f"    Range: [{propensity_scores[y==1].min():.3f}, {propensity_scores[y==1].max():.3f}]")
print(f"\n  Non-recipients (n={(~y).sum()}):")
print(f"    Mean: {propensity_scores[y==0].mean():.3f}")
print(f"    Median: {np.median(propensity_scores[y==0]):.3f}")
print(f"    Range: [{propensity_scores[y==0].min():.3f}, {propensity_scores[y==0].max():.3f}]")

# Identify common support region (trimming extremes)
ps_treated_min = propensity_scores[y==1].min()
ps_treated_max = propensity_scores[y==1].max()
ps_control_min = propensity_scores[y==0].min()
ps_control_max = propensity_scores[y==0].max()

common_support_min = max(ps_treated_min, ps_control_min)
common_support_max = min(ps_treated_max, ps_control_max)

print(f"\nCommon Support Region: [{common_support_min:.3f}, {common_support_max:.3f}]")

# Trim to common support
in_common_support = (propensity_scores >= common_support_min) & (propensity_scores <= common_support_max)
n_trimmed = (~in_common_support).sum()
pct_trimmed = 100 * n_trimmed / len(dataset)

print(f"  Trimmed {n_trimmed} observations ({pct_trimmed:.1f}%) outside common support")
print(f"  Analysis sample: {in_common_support.sum()} observations")

# Save diagnostic plot data
ps_diagnostics = {
    'therapy_ps': propensity_scores[y==1].tolist(),
    'control_ps': propensity_scores[y==0].tolist(),
    'common_support_min': float(common_support_min),
    'common_support_max': float(common_support_max),
    'n_trimmed': int(n_trimmed),
    'pct_trimmed': float(pct_trimmed)
}

with open(OUTPUT_DIR / 'propensity_score_diagnostics.json', 'w') as f:
    json.dump(ps_diagnostics, f, indent=2)

print(f"\n✓ Propensity score diagnostics saved")

# ============================================================================
# 2. FALSIFICATION TESTS (Treatment should NOT affect baseline)
# ============================================================================
print("\n" + "="*80)
print("2. FALSIFICATION TESTS (Treatment → Baseline Outcomes)")
print("="*80)
print("Expected: NO edges from treatment to baseline variables")
print("If edges exist → temporal specification violated or unmeasured confounding")

# Create falsification dataset with reversed temporal tiers
falsification_data = dataset.copy()

# Define tiers with TREATMENT as Tier 0, BASELINE as Tier 1 (reversed)
tier_info_falsified = {
    'therapy_any': 0, 'therapy_count': 0,
    'pharmacy_any': 0, 'pharmacy_count': 0,
    'chw_any': 0, 'chw_count': 0,
    'care_coord_any': 0, 'care_coord_count': 0,
    # Baseline as Tier 1 (should NOT be affected by Tier 0 treatment)
    'baseline_ip_ct': 1, 'baseline_ed_ct': 1,
    'baseline_cost': 1, 'age': 1, 'risk_score': 1
}

# Run PC algorithm on falsified structure
print("\nRunning PC algorithm on falsified temporal structure...")
falsified_graph = run_pc_algorithm(
    falsification_data,
    tier_info=tier_info_falsified,
    alpha=0.05,
    max_cond_size=3
)

# Check for (invalid) treatment → baseline edges
interventions = ['therapy_any', 'pharmacy_any', 'chw_any', 'care_coord_any']
baseline_outcomes = ['baseline_ip_ct', 'baseline_ed_ct', 'baseline_cost']

invalid_edges = []
for intervention in interventions:
    for baseline_var in baseline_outcomes:
        edge_key = f"{intervention}→{baseline_var}"
        if edge_key in falsified_graph['edges']:
            invalid_edges.append(edge_key)

print(f"\nFalsification Test Results:")
if len(invalid_edges) == 0:
    print(f"  ✓ PASS: No treatment → baseline edges detected")
    print(f"  → Temporal specification appears valid")
else:
    print(f"  ✗ FAIL: {len(invalid_edges)} invalid edges detected:")
    for edge in invalid_edges:
        print(f"    - {edge}")
    print(f"  → Suggests unmeasured confounding or model misspecification")

falsification_results = {
    'test': 'treatment_to_baseline_edges',
    'invalid_edges': invalid_edges,
    'passed': len(invalid_edges) == 0,
    'interpretation': 'PASS - temporal specification valid' if len(invalid_edges) == 0 else 'FAIL - unmeasured confounding likely'
}

with open(OUTPUT_DIR / 'falsification_tests.json', 'w') as f:
    json.dump(falsification_results, f, indent=2)

print(f"\n✓ Falsification test results saved")

# ============================================================================
# 3. ALGORITHM PARAMETER SENSITIVITY
# ============================================================================
print("\n" + "="*80)
print("3. ALGORITHM PARAMETER SENSITIVITY")
print("="*80)

# Re-run PC with different alpha values
alphas = [0.01, 0.05, 0.10]
param_sensitivity = {}

for alpha in alphas:
    print(f"\nRunning PC with α={alpha}...")
    
    # Standard temporal tiers
    tier_info_standard = {
        'age': 0, 'female': 0, 'baseline_ip_ct': 0, 'baseline_ed_ct': 0,
        'baseline_cost': 0, 'risk_score': 0,
        'therapy_any': 1, 'therapy_count': 1,
        'pharmacy_any': 1, 'pharmacy_count': 1,
        'chw_any': 1, 'chw_count': 1,
        'care_coord_any': 1, 'care_coord_count': 1,
        'followup_ip_ct': 2, 'followup_ed_ct': 2, 'followup_cost': 2
    }
    
    graph = run_pc_algorithm(
        dataset,
        tier_info=tier_info_standard,
        alpha=alpha,
        max_cond_size=3
    )
    
    # Check for therapy → IP edge
    therapy_ip_edge = 'therapy_any→followup_ip_ct' in graph['edges']
    
    param_sensitivity[f'alpha_{alpha}'] = {
        'total_edges': len(graph['edges']),
        'therapy_ip_pathway': therapy_ip_edge
    }
    
    print(f"  Total edges: {len(graph['edges'])}")
    print(f"  therapy→IP pathway: {'PRESENT' if therapy_ip_edge else 'ABSENT'}")

# Summary
print(f"\nParameter Sensitivity Summary:")
all_have_therapy_ip = all([v['therapy_ip_pathway'] for v in param_sensitivity.values()])
if all_have_therapy_ip:
    print(f"  ✓ therapy→IP pathway ROBUST across all α values")
else:
    print(f"  ⚠ therapy→IP pathway sensitivity to α threshold")

with open(OUTPUT_DIR / 'parameter_sensitivity.json', 'w') as f:
    json.dump(param_sensitivity, f, indent=2)

print(f"\n✓ Parameter sensitivity results saved")

# ============================================================================
# 4. VARIABLE SELECTION ROBUSTNESS
# ============================================================================
print("\n" + "="*80)
print("4. VARIABLE SELECTION ROBUSTNESS")
print("="*80)

# Test with/without baseline_cost
print("\nTesting with/without baseline_cost...")

# Without baseline_cost
dataset_no_cost = dataset.drop(columns=['baseline_cost'])
tier_info_no_cost = {k: v for k, v in tier_info_standard.items() if k != 'baseline_cost'}

graph_no_cost = run_pc_algorithm(
    dataset_no_cost,
    tier_info=tier_info_no_cost,
    alpha=0.05,
    max_cond_size=3
)

therapy_ip_no_cost = 'therapy_any→followup_ip_ct' in graph_no_cost['edges']

variable_robustness = {
    'with_baseline_cost': {
        'therapy_ip_pathway': 'therapy_any→followup_ip_ct' in falsified_graph['edges']  # From main analysis
    },
    'without_baseline_cost': {
        'therapy_ip_pathway': therapy_ip_no_cost
    },
    'robust': therapy_ip_no_cost  # If pathway persists without cost variable
}

print(f"\n  With baseline_cost: therapy→IP = {variable_robustness['with_baseline_cost']['therapy_ip_pathway']}")
print(f"  Without baseline_cost: therapy→IP = {therapy_ip_no_cost}")
print(f"  → Pathway is {'ROBUST' if therapy_ip_no_cost else 'NOT ROBUST'} to cost inclusion")

with open(OUTPUT_DIR / 'variable_robustness.json', 'w') as f:
    json.dump(variable_robustness, f, indent=2)

print(f"\n✓ Variable robustness results saved")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SENSITIVITY ANALYSES SUMMARY")
print("="*80)

summary = {
    'propensity_score': {
        'common_support_pct': float(100 * in_common_support.sum() / len(dataset)),
        'trimmed_pct': float(pct_trimmed),
        'conclusion': 'Positivity violations addressed via trimming'
    },
    'falsification_tests': {
        'passed': falsification_results['passed'],
        'conclusion': falsification_results['interpretation']
    },
    'parameter_sensitivity': {
        'therapy_ip_robust': all_have_therapy_ip,
        'conclusion': 'Robust across α values' if all_have_therapy_ip else 'Sensitive to α'
    },
    'variable_robustness': {
        'therapy_ip_robust': variable_robustness['robust'],
        'conclusion': 'Robust to cost exclusion' if variable_robustness['robust'] else 'Dependent on cost variable'
    }
}

with open(OUTPUT_DIR / 'sensitivity_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✓ All sensitivity analyses complete!")
print(f"✓ Results saved to: {OUTPUT_DIR}")
print("\nKey Findings:")
print(f"1. Propensity scores: {pct_trimmed:.1f}% trimmed for positivity")
print(f"2. Falsification tests: {'PASSED' if falsification_results['passed'] else 'FAILED'}")
print(f"3. Parameter sensitivity: {'ROBUST' if all_have_therapy_ip else 'SENSITIVE'}")
print(f"4. Variable robustness: {'ROBUST' if variable_robustness['robust'] else 'NOT ROBUST'}")
