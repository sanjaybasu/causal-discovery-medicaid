"""
Empirically Analyze Age and Temporal Thresholds for Manuscript

This script analyzes:
1. Age distribution among therapy recipients vs non-recipients
2. Empirical age thresholds (quantiles, median splits, optimal splits)
3. Temporal definition of "recent" hospitalizations based on baseline window
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from causal_discovery.data_loader_enhanced import (
    load_causal_dataset_enhanced_optimized,
    TemporalConfig,
)

RESULTS_DIR = REPO_ROOT / "results" / "causal_discovery_expanded"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_age_thresholds(dataset):
    """Empirically analyze age thresholds for therapy exposure."""
    print("\n" + "="*80)
    print("AGE THRESHOLD ANALYSIS")
    print("="*80)
    
    # Overall age distribution
    print(f"\nOverall Age Statistics:")
    print(f"  Mean: {dataset['age'].mean():.1f} years")
    print(f"  Median: {dataset['age'].median():.1f} years")
    print(f"  SD: {dataset['age'].std():.1f} years")
    print(f"  Range: {dataset['age'].min():.1f} - {dataset['age'].max():.1f} years")
    print(f"  Q1 (25th): {dataset['age'].quantile(0.25):.1f} years")
    print(f"  Q3 (75th): {dataset['age'].quantile(0.75):.1f} years")
    
    # Age by therapy exposure
    if 'therapy_any' in dataset.columns:
        therapy_yes = dataset[dataset['therapy_any'] == 1]
        therapy_no = dataset[dataset['therapy_any'] == 0]
        
        print(f"\nAge Among Therapy Recipients (n={len(therapy_yes)}):")
        print(f"  Mean: {therapy_yes['age'].mean():.1f} years")
        print(f"  Median: {therapy_yes['age'].median():.1f} years")
        print(f"  SD: {therapy_yes['age'].std():.1f} years")
        
        print(f"\nAge Among Non-Recipients (n={len(therapy_no)}):")
        print(f"  Mean: {therapy_no['age'].mean():.1f} years")
        print(f"  Median: {therapy_no['age'].median():.1f} years")
        print(f"  SD: {therapy_no['age'].std():.1f} years")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(therapy_yes['age'], therapy_no['age'])
        print(f"\nIndependent t-test:")
        print(f"  Mean difference: {therapy_yes['age'].mean() - therapy_no['age'].mean():.1f} years")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        
        # Quantile analysis
        print(f"\nTherapy Exposure by Age Quantile:")
        dataset['age_quartile'] = pd.qcut(dataset['age'], q=4, labels=['Q1 (youngest)', 'Q2', 'Q3', 'Q4 (oldest)'])
        exposure_by_quartile = dataset.groupby('age_quartile')['therapy_any'].agg(['sum', 'count', 'mean'])
        exposure_by_quartile['percent'] = exposure_by_quartile['mean'] * 100
        print(exposure_by_quartile[['sum', 'count', 'percent']])
        
        # Try common age thresholds
        print(f"\nTherapy Exposure by Common Age Thresholds:")
        for threshold in [50, 55, 60, 65, 70]:
            older = dataset[dataset['age'] >= threshold]
            younger = dataset[dataset['age'] < threshold]
            print(f"  Age ≥{threshold}: {older['therapy_any'].mean()*100:.1f}% (n={len(older)})")
            print(f"  Age <{threshold}: {younger['therapy_any'].mean()*100:.1f}% (n={len(younger)})")
            
            # Chi-square test
            contingency = pd.crosstab(dataset['age'] >= threshold, dataset['therapy_any'])
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            print(f"    χ²={chi2:.2f}, p={p_val:.4f}")
        
        # Optimal split analysis (maximize chi-square)
        print(f"\nSearching for Optimal Age Split:")
        best_age = None
        best_chi2 = 0
        best_p = 1.0
        
        for age in range(30, 80, 5):
            if len(dataset[dataset['age'] >= age]) > 100 and len(dataset[dataset['age'] < age]) > 100:
                contingency = pd.crosstab(dataset['age'] >= age, dataset['therapy_any'])
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_p = p_val
                    best_age = age
        
        if best_age:
            print(f"  Optimal split: Age {best_age} years")
            print(f"  χ²={best_chi2:.2f}, p={best_p:.4f}")
            older = dataset[dataset['age'] >= best_age]
            younger = dataset[dataset['age'] < best_age]
            print(f"  Age ≥{best_age}: {older['therapy_any'].mean()*100:.1f}% exposed")
            print(f"  Age <{best_age}: {younger['therapy_any'].mean()*100:.1f}% exposed")
    
    return {
        'mean_age_overall': float(dataset['age'].mean()),
        'median_age_overall': float(dataset['age'].median()),
        'mean_age_therapy': float(therapy_yes['age'].mean()) if 'therapy_any' in dataset.columns else None,
        'mean_age_no_therapy': float(therapy_no['age'].mean()) if 'therapy_any' in dataset.columns else None,
        'optimal_age_threshold': best_age if 'therapy_any' in dataset.columns else None,
    }

def analyze_baseline_window():
    """Define 'recent' hospitalizations based on temporal config."""
    print("\n" + "="*80)
    print("TEMPORAL WINDOW DEFINITION")
    print("="*80)
    
    config = TemporalConfig(
        baseline_months=6,
        followup_months=6,
        intervention_buffer_days=30,
    )
    
    print(f"\nBaseline Period: {config.baseline_months} months prior to activation")
    print(f"  → 'Recent' hospitalizations = within {config.baseline_months} months")
    print(f"\nIntervention Buffer: {config.intervention_buffer_days} days post-activation")
    print(f"  → Mitigates immortal time bias")
    print(f"\nFollow-up Period: {config.followup_months} months post-activation")
    
    return {
        'baseline_months': config.baseline_months,
        'recent_definition': f"within {config.baseline_months} months prior to activation",
        'buffer_days': config.intervention_buffer_days,
        'followup_months': config.followup_months,
    }

def main():
    """Run empirical threshold analysis."""
    print("="*80)
    print("EMPIRICAL THRESHOLD ANALYSIS FOR MANUSCRIPT")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    config = TemporalConfig(
        baseline_months=6,
        followup_months=6,
        intervention_buffer_days=30,
    )
    
    dataset, metadata = load_causal_dataset_enhanced_optimized(
        config=config,
        sample_size=5000,
    )
    
    # Analyze age thresholds
    age_results = analyze_age_thresholds(dataset)
    
    # Define temporal windows
    temporal_results = analyze_baseline_window()
    
    # Combine results
    results = {
        'age_analysis': age_results,
        'temporal_definitions': temporal_results,
    }
    
    # Save results
    output_file = RESULTS_DIR / 'empirical_thresholds.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print recommendations
    print("\nRECOMMENDATIONS FOR MANUSCRIPT:")
    print("-" * 80)
    
    if age_results.get('mean_age_therapy') and age_results.get('mean_age_no_therapy'):
        diff = age_results['mean_age_therapy'] - age_results['mean_age_no_therapy']
        print(f"\n1. AGE: Instead of 'older adults', report:")
        print(f"   'Therapy recipients were {abs(diff):.1f} years older on average")
        print(f"    ({age_results['mean_age_therapy']:.1f} vs {age_results['mean_age_no_therapy']:.1f} years)'")
        
    if age_results.get('optimal_age_threshold'):
        print(f"\n   OR use empirically-derived threshold:")
        print(f"   'adults age ≥{age_results['optimal_age_threshold']} years'")
    
    print(f"\n2. 'RECENT' HOSPITALIZATIONS:")
    print(f"   Replace 'recent' with '{temporal_results['recent_definition']}'")
    print(f"   This is empirically derived from the baseline assessment window")
    
    return results

if __name__ == '__main__':
    main()
