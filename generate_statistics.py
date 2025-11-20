"""
Generate Real Statistics for Causal Discovery Manuscript

This script calculates all real statistics from the actual analysis data
to replace illustrative placeholders in the manuscript.

Author: Sanjay Basu
Date: November 2025
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson
from statsmodels.stats.multitest import fdrcorrection

# Paths
DATA_DIR = Path('/Users/sanjaybasu/waymark-local/data/real_inputs')
RESULTS_DIR = Path('/Users/sanjaybasu/waymark-local/results/causal_discovery_expanded')
OUTPUT_DIR = Path('/Users/sanjaybasu/waymark-local/notebooks/causal_discovery_publication')

def load_analysis_data():
    """Load the data that was used in causal discovery analysis"""
    print("Loading analysis data...")
    
    # Import the data loader
    import sys
    sys.path.append('/Users/sanjaybasu/waymark-local/notebooks/causal_discovery')
    from data_loader_enhanced import load_and_prepare_data
    
    # Load same data used in analysis
    data, variable_info, member_data = load_and_prepare_data(
        data_dir=str(DATA_DIR),
        sample_size=5000,
        seed=42
    )
    
    return data, variable_info, member_data

def calculate_table1_demographics(member_data):
    """Calculate Table 1: Study Population Characteristics"""
    print("\nCalculating Table 1 demographics...")
    
    stats = {
        'n_total': len(member_data),
        'age_mean': member_data['age'].mean(),
        'age_std': member_data['age'].std(),
        'female_n': (member_data['gender'] == 'F').sum() if 'gender' in member_data else None,
        'female_pct': (member_data['gender'] == 'F').mean() * 100 if 'gender' in member_data else None,
        'risk_score_mean': member_data['risk_score'].mean() if 'risk_score' in member_data else None,
        'risk_score_std': member_data['risk_score'].std() if 'risk_score' in member_data else None,
        'baseline_ed_mean': member_data['baseline_ed'].mean(),
        'baseline_ed_std': member_data['baseline_ed'].std(),
        'baseline_ip_mean': member_data['baseline_ip'].mean(),
        'baseline_ip_std': member_data['baseline_ip'].std(),
        'baseline_cost_mean': member_data['baseline_cost'].mean() if 'baseline_cost' in member_data else None,
        'baseline_cost_std': member_data['baseline_cost'].std() if 'baseline_cost' in member_data else None,
        'followup_ed_mean': member_data['followup_ed'].mean(),
        'followup_ed_std': member_data['followup_ed'].std(),
        'followup_ip_mean': member_data['followup_ip'].mean(),
        'followup_ip_std': member_data['followup_ip'].std(),
        'followup_cost_mean': member_data['followup_cost'].mean() if 'followup_cost' in member_data else None,
        'followup_cost_std': member_data['followup_cost'].std() if 'followup_cost' in member_data else None,
    }
    
    # Intervention exposures
    if 'therapy' in member_data.columns:
        stats['therapy_n'] = member_data['therapy'].sum()
        stats['therapy_pct'] = (member_data['therapy'].sum() / len(member_data)) * 100
    if 'pharmacy' in member_data.columns:
        stats['pharmacy_n'] = member_data['pharmacy'].sum()
        stats['pharmacy_pct'] = (member_data['pharmacy'].sum() / len(member_data)) * 100
    if 'chw' in member_data.columns:
        stats['chw_n'] = member_data['chw'].sum()
        stats['chw_pct'] = (member_data['chw'].sum() / len(member_data)) * 100
    if 'care_coordination' in member_data.columns:
        stats['care_coord_n'] = member_data['care_coordination'].sum()
        stats['care_coord_pct'] = (member_data['care_coordination'].sum() / len(member_data)) * 100
    
    return stats

def calculate_effect_estimates(member_data):
    """Calculate intervention effect estimates using Poisson regression"""
    print("\nCalculating effect estimates via Poisson regression...")
    
    effects = {}
    
    # Therapy → IP admissions
    if 'therapy' in member_data.columns and 'followup_ip' in member_data.columns:
        try:
            X = pd.DataFrame({
                'intercept': 1,
                'therapy': member_data['therapy'],
                'age': member_data['age'],
                'baseline_ip': member_data['baseline_ip']
            })
            y = member_data['followup_ip']
            
            model = GLM(y, X, family=Poisson())
            results = model.fit()
            
            rr = np.exp(results.params['therapy'])
            ci_lower = np.exp(results.conf_int().loc['therapy', 0])
            ci_upper = np.exp(results.conf_int().loc['therapy', 1])
            pvalue = results.pvalues['therapy']
            
            effects['therapy_ip'] = {
                'rr': float(rr),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'pvalue': float(pvalue)
            }
        except Exception as e:
            print(f"  Warning: Could not estimate therapy→IP effect: {e}")
    
    # Pharmacy → Costs
    if 'pharmacy_count' in member_data.columns and 'followup_cost' in member_data.columns:
        try:
            X = pd.DataFrame({
                'intercept': 1,
                'pharmacy_count': member_data['pharmacy_count'],
                'age': member_data['age'],
                'baseline_cost': member_data.get('baseline_cost', 0)
            })
            y = member_data['followup_cost']
            
            model = GLM(y, X, family=Poisson())
            results = model.fit()
            
            rr = np.exp(results.params['pharmacy_count'])
            effects['pharmacy_cost'] = {
                'rr_per_contact': float(rr),
                'pvalue': float(results.pvalues['pharmacy_count'])
            }
        except Exception as e:
            print(f"  Warning: Could not estimate pharmacy→cost effect: {e}")
    
    # CHW → ED visits
    if 'chw_count' in member_data.columns and 'followup_ed' in member_data.columns:
        try:
            X = pd.DataFrame({
                'intercept': 1,
                'chw_count': member_data['chw_count'],
                'baseline_ed': member_data['baseline_ed']
            })
            y = member_data['followup_ed']
            
            model = GLM(y, X, family=Poisson())
            results = model.fit()
            
            rr = np.exp(results.params['chw_count'])
            effects['chw_ed'] = {
                'rr_per_contact': float(rr),
                'pvalue': float(results.pvalues['chw_count'])
            }
        except Exception as e:
            print(f"  Warning: Could not estimate CHW→ED effect: {e}")
    
    # Care coordination → ED visits
    if 'care_coordination' in member_data.columns and 'followup_ed' in member_data.columns:
        try:
            X = pd.DataFrame({
                'intercept': 1,
                'care_coordination': member_data['care_coordination'],
                'female': (member_data.get('gender') == 'F').astype(int) if 'gender' in member_data else 0,
                'baseline_ed': member_data['baseline_ed']
            })
            y = member_data['followup_ed']
            
            model = GLM(y, X, family=Poisson())
            results = model.fit()
            
            rr = np.exp(results.params['care_coordination'])
            ci_lower = np.exp(results.conf_int().loc['care_coordination', 0])
            ci_upper = np.exp(results.conf_int().loc['care_coordination', 1])
            
            effects['care_coord_ed'] = {
                'rr': float(rr),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'pvalue': float(results.pvalues['care_coordination'])
            }
        except Exception as e:
            print(f"  Warning: Could not estimate care coordination→ED effect: {e}")
    
    return effects

def calculate_evalues(effects):
    """Calculate E-values from effect estimates"""
    print("\nCalculating E-values...")
    
    def evalue_formula(rr):
        """E-value = RR + sqrt(RR * (RR - 1))"""
        if rr < 1:
            rr = 1 / rr  # Convert protective to harmful scale
        return rr + np.sqrt(rr * (rr - 1))
    
    evalues = {}
    
    # Therapy → IP
    if 'therapy_ip' in effects:
        rr = effects['therapy_ip']['rr']
        ci_lower = effects['therapy_ip']['ci_lower']
        evalues['therapy_ip'] = {
            'point_estimate': evalue_formula(rr),
            'ci_limit': evalue_formula(ci_lower) if rr < 1 else evalue_formula(effects['therapy_ip']['ci_upper'])
        }
    
    # Pharmacy → Cost (per 3 contacts, 75th-25th percentile)
    if 'pharmacy_cost' in effects:
        rr_per_contact = effects['pharmacy_cost']['rr_per_contact']
        rr_3contacts = rr_per_contact ** 3
        evalues['pharmacy_cost'] = {
            'point_estimate': evalue_formula(rr_3contacts),
'ci_limit': None  # Would need CI from regression
        }
    
    # CHW → ED (per 5 contacts)
    if 'chw_ed' in effects:
        rr_per_contact = effects['chw_ed']['rr_per_contact']
        rr_5contacts = rr_per_contact ** 5
        evalues['chw_ed'] = {
            'point_estimate': evalue_formula(rr_5contacts),
            'ci_limit': None
        }
    
    # Care coordination → ED
    if 'care_coord_ed' in effects:
        rr = effects['care_coord_ed']['rr']
        ci_lower = effects['care_coord_ed']['ci_lower']
        evalues['care_coord_ed'] = {
            'point_estimate': evalue_formula(rr),
            'ci_limit': evalue_formula(ci_lower) if rr < 1 else evalue_formula(effects['care_coord_ed']['ci_upper'])
        }
    
    return evalues

def extract_pvalues_from_analysis():
    """Extract p-values from actual PC algorithm conditional independence tests"""
    print("\nExtracting p-values from PC algorithm output...")
    
    # Load edge lists which should contain test statistics
    try:
        pc_edges = pd.read_csv(RESULTS_DIR / 'pc_edges_expanded.csv')
        
        # If p-values are in the CSV, extract them
        if 'pvalue' in pc_edges.columns:
            pvalues = pc_edges['pvalue'].values
            
            # Apply Benjamini-Hochberg FDR correction
            reject, pvals_corrected = fdrcorrection(pvalues, alpha=0.05)
            
            return {
                'n_tests': len(pvalues),
                'n_significant_nominal': (pvalues < 0.05).sum(),
                'n_significant_fdr': reject.sum(),
                'fdr_threshold': pvals_corrected[reject].max() if reject.sum() > 0 else None,
                'pvalues_sample': pvalues[:10].tolist()  # First 10 as example
            }
        else:
            print("  Warning: No p-values found in PC edge list")
            return None
            
    except Exception as e:
        print(f"  Warning: Could not extract p-values: {e}")
        return None

def main():
    """Generate all real statistics for manuscript"""
    print("=" * 60)
    print("GENERATING REAL STATISTICS FOR MANUSCRIPT")
    print("=" * 60)
    
    # Load data
    try:
        data, variable_info, member_data = load_analysis_data()
    except Exception as e:
        print(f"\nERROR: Could not load analysis data: {e}")
        print("This script requires the actual data files used in the analysis.")
        return
    
    # Calculate statistics
    results = {}
    
    # Table 1
    results['table1'] = calculate_table1_demographics(member_data)
    
    # Effect estimates
    results['effects'] = calculate_effect_estimates(member_data)
    
    # E-values
    results['evalues'] = calculate_evalues(results['effects'])
    
    # P-values and FDR
    results['pvalues_fdr'] = extract_pvalues_from_analysis()
    
    # Save results
    output_file = OUTPUT_DIR / 'real_statistics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"RESULTS SAVED TO: {output_file}")
    print("=" * 60)
    
    # Print summary
    print("\nSUMMARY OF REAL STATISTICS:")
    print(f"\nTable 1 Demographics:")
    print(f"  N = {results['table1']['n_total']}")
    print(f"  Age: {results['table1']['age_mean']:.1f} ± {results['table1']['age_std']:.1f}")
    if results['table1'].get('female_pct'):
        print(f"  Female: {results['table1']['female_pct']:.1f}%")
    print(f"  Baseline ED: {results['table1']['baseline_ed_mean']:.2f} ± {results['table1']['baseline_ed_std']:.2f}")
    print(f"  Baseline IP: {results['table1']['baseline_ip_mean']:.2f} ± {results['table1']['baseline_ip_std']:.2f}")
    
    print(f"\nEffect Estimates:")
    for effect_name, effect_data in results['effects'].items():
        print(f"  {effect_name}:")
        if 'rr' in effect_data:
            print(f"    RR = {effect_data['rr']:.3f} (p = {effect_data['pvalue']:.4f})")
        elif 'rr_per_contact' in effect_data:
            print(f"    RR per contact = {effect_data['rr_per_contact']:.3f} (p = {effect_data['pvalue']:.4f})")
    
    print(f"\nE-values:")
    for evalue_name, evalue_data in results['evalues'].items():
        print(f"  {evalue_name}: {evalue_data['point_estimate']:.2f}")
    
    if results['pvalues_fdr']:
        print(f"\nMultiple Testing (Benjamini-Hochberg FDR):")
        print(f"  Total tests: {results['pvalues_fdr']['n_tests']}")
        print(f"  Significant (nominal α=0.05): {results['pvalues_fdr']['n_significant_nominal']}")
        print(f"  Significant (FDR q=0.05): {results['pvalues_fdr']['n_significant_fdr']}")
    
    print("\n" + "=" * 60)
    print("Use these real statistics to update Tables 1-5 in manuscript")
    print("=" * 60)

if __name__ == '__main__':
    main()
