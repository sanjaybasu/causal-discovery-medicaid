"""
Extract Real Statistics for Manuscript Tables
Uses the exact same data loading approach as run_expanded_analysis.py
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path (same as run_expanded_analysis.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from causal_discovery.data_loader_enhanced import (
    load_causal_dataset_enhanced_optimized,
    TemporalConfig,
)

# Output
OUTPUT_DIR = Path('/Users/sanjaybasu/waymark-local/notebooks/causal_discovery_publication')

def main():
    """Extract real statistics from the 5,000-member dataset."""
    print("=" * 80)
    print("EXTRACTING REAL STATISTICS FROM ANALYSIS DATA")
    print("=" * 80)
    
    # 1. Load EXACT same data as run_expanded_analysis.py
    print("\n1. Loading data (same as run_expanded_analysis.py)...")
    config = TemporalConfig(
        baseline_months=6,
        followup_months=6,
        intervention_buffer_days=30,
    )
    
    dataset, metadata = load_causal_dataset_enhanced_optimized(
        config=config,
        sample_size=5000,  # Same as analysis
    )
    
    print(f"   ✓ Loaded {dataset.shape[0]} members, {dataset.shape[1]} variables")
    print(f"   ✓ Metadata: {metadata}")
    
    # 2. Select EXACT same variables as analysis
    baseline_vars = [
        "age",
        "gender_female",
        "risk_score",
        "baseline_ed_ct",
        "baseline_ip_ct",
        "baseline_total_paid",
    ]
    
    treatment_vars = [
        "therapy_any",
        "therapy_count",
        "pharmacy_any",
        "pharmacy_count",
        "chw_any",
        "chw_count",
        "care_coord_any",
        "care_coord_count",
    ]
    
    outcome_vars = [
        "followup_ed_ct",
        "followup_ip_ct",
        "followup_total_paid",
    ]
    
    all_vars = baseline_vars + treatment_vars + outcome_vars
    selected_vars = [v for v in all_vars if v in dataset.columns]
    analysis_data = dataset[selected_vars].copy()
    analysis_data = analysis_data.dropna()
    
    print(f"   ✓ Analysis dataset: {analysis_data.shape[0]} members after dropna")
    
    # 3. Calculate Table 1 Demographics
    print("\n2. Calculating Table 1 (Demographics)...")
    
    stats = {}
    
    # Sample size
    stats['n_total'] = len(analysis_data)
    
    # Age
    if 'age' in analysis_data.columns:
        stats['age_mean'] = float(analysis_data['age'].mean())
        stats['age_std'] = float(analysis_data['age'].std())
    
    # Gender
    if 'gender_female' in analysis_data.columns:
        stats['female_n'] = int(analysis_data['gender_female'].sum())
        stats['female_pct'] = float(analysis_data['gender_female'].mean() * 100)
    
    # Risk score
    if 'risk_score' in analysis_data.columns:
        stats['risk_score_mean'] = float(analysis_data['risk_score'].mean())
        stats['risk_score_std'] = float(analysis_data['risk_score'].std())
    
    # Baseline utilization
    for var in ['baseline_ed_ct', 'baseline_ip_ct', 'baseline_total_paid']:
        if var in analysis_data.columns:
            stats[f'{var}_mean'] = float(analysis_data[var].mean())
            stats[f'{var}_std'] = float(analysis_data[var].std())
    
    # Followup utilization
    for var in ['followup_ed_ct', 'followup_ip_ct', 'followup_total_paid']:
        if var in analysis_data.columns:
            stats[f'{var}_mean'] = float(analysis_data[var].mean())
            stats[f'{var}_std'] = float(analysis_data[var].std())
    
    # Intervention exposure
    for var in ['therapy_any', 'pharmacy_any', 'chw_any', 'care_coord_any']:
        if var in analysis_data.columns:
            stats[f'{var}_n'] = int(analysis_data[var].sum())
            stats[f'{var}_pct'] = float(analysis_data[var].mean() * 100)
    
    print("\n   Demographics:")
    print(f"     N = {stats['n_total']}")
    print(f"     Age: {stats.get('age_mean', 0):.1f} ± {stats.get('age_std', 0):.1f}")
    print(f"     Female: {stats.get('female_pct', 0):.1f}% (n={stats.get('female_n', 0)})")
    print(f"     Risk Score: {stats.get('risk_score_mean', 0):.2f} ± {stats.get('risk_score_std', 0):.2f}")
    print(f"     Baseline ED: {stats.get('baseline_ed_ct_mean', 0):.2f} ± {stats.get('baseline_ed_ct_std', 0):.2f}")
    print(f"     Baseline IP: {stats.get('baseline_ip_ct_mean', 0):.2f} ± {stats.get('baseline_ip_ct_std', 0):.2f}")
    
    print("\n   Intervention Exposure:")
    if 'therapy_any_pct' in stats:
        print(f"     Therapy: {stats['therapy_any_pct']:.1f}% (n={stats['therapy_any_n']})")
    if 'pharmacy_any_pct' in stats:
        print(f"     Pharmacy: {stats['pharmacy_any_pct']:.1f}% (n={stats['pharmacy_any_n']})")
    if 'chw_any_pct' in stats:
        print(f"     CHW: {stats['chw_any_pct']:.1f}% (n={stats['chw_any_n']})")
    if 'care_coord_any_pct' in stats:
        print(f"     Care Coordination: {stats['care_coord_any_pct']:.1f}% (n={stats['care_coord_any_n']})")
    
    # 4. Calculate simple effect estimates
    print("\n3. Calculating approximate effect sizes (for E-value bounds)...")
    
    effects = {}
    
    # Therapy → IP (compare therapy=1 vs therapy=0 on followup_ip_ct)
    if 'therapy_any' in analysis_data.columns and 'followup_ip_ct' in analysis_data.columns:
        therapy_yes = analysis_data[analysis_data['therapy_any'] == 1]['followup_ip_ct'].mean()
        therapy_no = analysis_data[analysis_data['therapy_any'] == 0]['followup_ip_ct'].mean()
        if therapy_no > 0:
            effects['therapy_ip_rr'] = float(therapy_yes / therapy_no)
            print(f"     Therapy→IP: RR ≈ {effects['therapy_ip_rr']:.3f}")
    
    # CHW → ED
    if 'chw_any' in analysis_data.columns and 'followup_ed_ct' in analysis_data.columns:
        chw_yes = analysis_data[analysis_data['chw_any'] == 1]['followup_ed_ct'].mean()
        chw_no = analysis_data[analysis_data['chw_any'] == 0]['followup_ed_ct'].mean()
        if chw_no > 0:
            effects['chw_ed_rr'] = float(chw_yes / chw_no)
            print(f"     CHW→ED: RR ≈ {effects['chw_ed_rr']:.3f}")
    
    # Care Coordination → ED
    if 'care_coord_any' in analysis_data.columns and 'followup_ed_ct' in analysis_data.columns:
        cc_yes = analysis_data[analysis_data['care_coord_any'] == 1]['followup_ed_ct'].mean()
        cc_no = analysis_data[analysis_data['care_coord_any'] == 0]['followup_ed_ct'].mean()
        if cc_no > 0:
            effects['care_coord_ed_rr'] = float(cc_yes / cc_no)
            print(f"     Care Coord→ED: RR ≈ {effects['care_coord_ed_rr']:.3f}")
    
    # 5. Calculate E-values from effect estimates
    print("\n4. Calculating E-values...")
    
    def calculate_evalue(rr):
        """E-value formula: RR + sqrt(RR * (RR - 1))"""
        if rr < 1:
            rr = 1 / rr  # Convert protective to harmful scale
        return rr + np.sqrt(rr * (rr - 1))
    
    evalues = {}
    for effect_name, rr in effects.items():
        evalue = calculate_evalue(rr)
        evalues[effect_name + '_evalue'] = float(evalue)
        print(f"     {effect_name}: E-value = {evalue:.2f}")
    
    # 6. Save all results
    results = {
        'table1_demographics': stats,
        'effect_estimates': effects,
        'evalues': evalues,
        'metadata': {
            'n_members_analyzed': stats['n_total'],
            'data_loading': 'load_causal_dataset_enhanced_optimized',
            'sample_size_requested': 5000,
            'config': {
                'baseline_months': 6,
                'followup_months': 6,
                'intervention_buffer_days': 30,
            }
        }
    }
    
    output_file = OUTPUT_DIR / 'real_statistics_extracted.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"✓ REAL STATISTICS SAVED TO: {output_file}")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review real_statistics_extracted.json")
    print("2. Update manuscript tables with these real numbers")
    print("3. Numbers are from actual 5,000-member analysis dataset")
    
    return results

if __name__ == '__main__':
    results = main()
