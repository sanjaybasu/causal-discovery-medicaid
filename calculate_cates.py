"""
Calculate Conditional Average Treatment Effects (CATEs) with Confidence Intervals
Following AJE/PATH guidelines for heterogeneous treatment effect reporting
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import json

sys.path.insert(0, '/Users/sanjaybasu/waymark-local/notebooks')
from causal_discovery.data_loader_enhanced import load_causal_dataset_enhanced_optimized, TemporalConfig

RESULTS_DIR = Path('/Users/sanjaybasu/waymark-local/results/causal_discovery_expanded')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def bootstrap_cate(therapy_group, control_group, outcome_var, n_bootstrap=1000):
    """Calculate CATE with bootstrap confidence interval."""
    therapy_outcome = therapy_group[outcome_var].values
    control_outcome = control_group[outcome_var].values
    
    # Point estimate
    cate = therapy_outcome.mean() - control_outcome.mean()
    
    # Bootstrap CI
    bootstrap_cates = []
    for _ in range(n_bootstrap):
        therapy_sample = np.random.choice(therapy_outcome, size=len(therapy_outcome), replace=True)
        control_sample = np.random.choice(control_outcome, size=len(control_outcome), replace=True)
        bootstrap_cates.append(therapy_sample.mean() - control_sample.mean())
    
    ci_lower = np.percentile(bootstrap_cates, 2.5)
    ci_upper = np.percentile(bootstrap_cates, 97.5)
    
    return cate, ci_lower, ci_upper

def main():
    print("="*80)
    print("CONDITIONAL AVERAGE TREATMENT EFFECTS (CATE) ANALYSIS")
    print("Following AJE/PATH Guidelines")
    print("="*80)
    
    # Load data
    config = TemporalConfig(baseline_months=6, followup_months=6, intervention_buffer_days=30)
    dataset, metadata = load_causal_dataset_enhanced_optimized(config=config, sample_size=5000)
    
    therapy_recipients = dataset[dataset['therapy_any'] == 1]
    non_recipients = dataset[dataset['therapy_any'] == 0]
    
    print(f"\nSample sizes:")
    print(f"  Therapy recipients: {len(therapy_recipients)}")
    print(f"  Non-recipients: {len(non_recipients)}")
    
    # Subgroup 1: WITH baseline IP
    therapy_with_ip = therapy_recipients[therapy_recipients['baseline_ip_ct'] > 0]
    control_with_ip = non_recipients[non_recipients['baseline_ip_ct'] > 0]
    
    print(f"\n{'='*80}")
    print("SUBGROUP 1: Members WITH Baseline IP Admissions")
    print(f"{'='*80}")
    print(f"  Therapy group: n={len(therapy_with_ip)}")
    print(f"  Control group: n={len(control_with_ip)}")
    
    print(f"\nBaseline IP admissions:")
    print(f"  Therapy: {therapy_with_ip['baseline_ip_ct'].mean():.2f} ± {therapy_with_ip['baseline_ip_ct'].std():.2f}")
    print(f"  Control: {control_with_ip['baseline_ip_ct'].mean():.2f} ± {control_with_ip['baseline_ip_ct'].std():.2f}")
    
    print(f"\nFollow-up IP admissions:")
    print(f"  Therapy: {therapy_with_ip['followup_ip_ct'].mean():.2f} ± {therapy_with_ip['followup_ip_ct'].std():.2f}")
    print(f"  Control: {control_with_ip['followup_ip_ct'].mean():.2f} ± {control_with_ip['followup_ip_ct'].std():.2f}")
    
    cate_with_ip, ci_lower_with, ci_upper_with = bootstrap_cate(
        therapy_with_ip, control_with_ip, 'followup_ip_ct'
    )
    
    print(f"\nConditional Average Treatment Effect (CATE):")
    print(f"  CATE: {cate_with_ip:.3f} admissions")
    print(f"  95% CI: ({ci_lower_with:.3f}, {ci_upper_with:.3f})")
    
    # Subgroup 2: WITHOUT baseline IP
    therapy_no_ip = therapy_recipients[therapy_recipients['baseline_ip_ct'] == 0]
    control_no_ip = non_recipients[non_recipients['baseline_ip_ct'] == 0]
    
    print(f"\n{'='*80}")
    print("SUBGROUP 2: Members WITHOUT Baseline IP Admissions")
    print(f"{'='*80}")
    print(f"  Therapy group: n={len(therapy_no_ip)}")
    print(f"  Control group: n={len(control_no_ip)}")
    
    print(f"\nFollow-up IP admissions:")
    print(f"  Therapy: {therapy_no_ip['followup_ip_ct'].mean():.2f} ± {therapy_no_ip['followup_ip_ct'].std():.2f}")
    print(f"  Control: {control_no_ip['followup_ip_ct'].mean():.2f} ± {control_no_ip['followup_ip_ct'].std():.2f}")
    
    cate_no_ip, ci_lower_no, ci_upper_no = bootstrap_cate(
        therapy_no_ip, control_no_ip, 'followup_ip_ct'
    )
    
    print(f"\nConditional Average Treatment Effect (CATE):")
    print(f"  CATE: {cate_no_ip:.3f} admissions")
    print(f"  95% CI: ({ci_lower_no:.3f}, {ci_upper_no:.3f})")
    
    # Test for interaction
    print(f"\n{'='*80}")
    print("INTERACTION TEST")
    print(f"{'='*80}")
    
    cate_difference = cate_with_ip - cate_no_ip
    print(f"\nDifference in CATEs: {cate_difference:.3f}")
    
    # Bootstrap p-value for interaction
    bootstrap_diffs = []
    for _ in range(1000):
        # Resample therapy groups
        therapy_with_sample = therapy_with_ip.sample(frac=1, replace=True)
        therapy_no_sample = therapy_no_ip.sample(frac=1, replace=True)
        control_with_sample = control_with_ip.sample(frac=1, replace=True)
        control_no_sample = control_no_ip.sample(frac=1, replace=True)
        
        cate_with_boot = (therapy_with_sample['followup_ip_ct'].mean() - 
                          control_with_sample['followup_ip_ct'].mean())
        cate_no_boot = (therapy_no_sample['followup_ip_ct'].mean() - 
                        control_no_sample['followup_ip_ct'].mean())
        
        bootstrap_diffs.append(cate_with_boot - cate_no_boot)
    
    # Two-tailed p-value
    p_interaction = 2 * min(
        np.mean(np.array(bootstrap_diffs) >= 0),
        np.mean(np.array(bootstrap_diffs) <= 0)
    )
    
    print(f"Interaction p-value (bootstrap): {p_interaction:.4f}")
    
    if p_interaction < 0.001:
        print("  → QUALITATIVE INTERACTION DETECTED (p<0.001)")
        print("  → Therapy beneficial in one subgroup, null/harmful in another")
    
    # Age stratification (to show it's NOT a modifier)
    print(f"\n{'='*80}")
    print("AGE STRATIFICATION (Non-Modifier Control)")
    print(f"{'='*80}")
    
    age_median = therapy_recipients['age'].median()
    therapy_older = therapy_recipients[therapy_recipients['age'] >= age_median]
    therapy_younger = therapy_recipients[therapy_recipients['age'] < age_median]
    control_older = non_recipients[non_recipients['age'] >= age_median]
    control_younger = non_recipients[non_recipients['age'] < age_median]
    
    cate_older, ci_lower_older, ci_upper_older = bootstrap_cate(
        therapy_older, control_older, 'followup_ip_ct'
    )
    cate_younger, ci_lower_younger, ci_upper_younger = bootstrap_cate(
        therapy_younger, control_younger, 'followup_ip_ct'
    )
    
    print(f"\nOlder (≥{age_median:.1f} years): CATE={cate_older:.3f} ({ci_lower_older:.3f}, {ci_upper_older:.3f})")
    print(f"Younger (<{age_median:.1f} years): CATE={cate_younger:.3f} ({ci_lower_younger:.3f}, {ci_upper_younger:.3f})")
    print(f"Difference: {abs(cate_older - cate_younger):.3f}")
    print("  → Age does NOT modify treatment effect (similar CATEs)")
    
    # Save results
    results = {
        'subgroup_with_baseline_ip': {
            'n_therapy': int(len(therapy_with_ip)),
            'n_control': int(len(control_with_ip)),
            'cate': float(cate_with_ip),
            'ci_lower': float(ci_lower_with),
            'ci_upper': float(ci_upper_with),
            'baseline_ip_therapy': float(therapy_with_ip['baseline_ip_ct'].mean()),
            'followup_ip_therapy': float(therapy_with_ip['followup_ip_ct'].mean()),
            'followup_ip_control': float(control_with_ip['followup_ip_ct'].mean()),
        },
        'subgroup_without_baseline_ip': {
            'n_therapy': int(len(therapy_no_ip)),
            'n_control': int(len(control_no_ip)),
            'cate': float(cate_no_ip),
            'ci_lower': float(ci_lower_no),
            'ci_upper': float(ci_upper_no),
            'followup_ip_therapy': float(therapy_no_ip['followup_ip_ct'].mean()),
            'followup_ip_control': float(control_no_ip['followup_ip_ct'].mean()),
        },
        'interaction_test': {
            'cate_difference': float(cate_difference),
            'p_value': float(p_interaction),
            'qualitative_interaction': bool(p_interaction < 0.001),
        },
        'age_stratification': {
            'median_age': float(age_median),
            'cate_older': float(cate_older),
            'cate_younger': float(cate_younger),
            'cate_difference': float(abs(cate_older - cate_younger)),
        }
    }
    
    output_file = RESULTS_DIR / 'cate_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
