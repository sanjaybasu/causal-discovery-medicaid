"""Test causal discovery algorithms on synthetic data with known structure."""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from causal_discovery.algorithms import PCAlgorithm, GESAlgorithm


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic data with known causal structure.
    
    True DAG:
        X1 -> X3 -> Y
        X2 -> X3
        X1 -> Y
    
    Where:
        X1, X2 = exogenous baseline variables
        X3 = mediator (treatment)
        Y = outcome
    """
    np.random.seed(seed)
    
    # Exogenous variables
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    
    # Mediator: X3 = 0.5*X1 + 0.5*X2 + noise
    X3 = 0.5 * X1 + 0.5 * X2 + np.random.randn(n_samples) * 0.5
    
    # Outcome: Y = 0.6*X1 + 0.7*X3 + noise
    Y = 0.6 * X1 + 0.7 * X3 + np.random.randn(n_samples) * 0.5
    
    return pd.DataFrame({
        "X1": X1,
        "X2": X2,
        "X3": X3,
        "Y": Y,
    })


def test_pc_algorithm():
    """Test PC algorithm on synthetic data."""
    print("=" * 60)
    print("Testing PC Algorithm")
    print("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_samples=1000)
    
    # Define temporal tiers (to constrain search space)
    temporal_tiers = [
        ["X1", "X2"],  # Baseline
        ["X3"],         # Treatment
        ["Y"],          # Outcome
    ]
    
    # Run PC
    pc = PCAlgorithm(alpha=0.05, max_conditioning_set_size=2, temporal_tiers=temporal_tiers)
    graph = pc.fit(data)
    
    print(f"\nLearned edges:")
    for edge in sorted(graph.edges):
        print(f"  {edge[0]} -> {edge[1]}")
    
    print(f"\nUndirected edges:")
    for edge in sorted(graph.undirected_edges):
        print(f"  {edge[0]} - {edge[1]}")
    
    # Check if key edges are recovered
    edges_set = set(graph.edges)
    expected_edges = {("X1", "X3"), ("X2", "X3"), ("X3", "Y"), ("X1", "Y")}
    
    print(f"\n✓ Expected edges: {expected_edges}")
    print(f"✓ Learned edges: {edges_set}")
    
    recovered = expected_edges & edges_set
    print(f"\n✓ Correctly recovered: {len(recovered)}/{len(expected_edges)} edges")
    for edge in recovered:
        print(f"    {edge[0]} -> {edge[1]}")
    
    return graph


def test_ges_algorithm():
    """Test GES algorithm on synthetic data."""
    print("\n" + "=" * 60)
    print("Testing GES Algorithm")
    print("=" * 60)
    
    # Generate data
    data = generate_synthetic_data(n_samples=1000)
    
    # Define temporal tiers
    temporal_tiers = [
        ["X1", "X2"],
        ["X3"],
        ["Y"],
    ]
    
    # Run GES
    ges = GESAlgorithm(temporal_tiers=temporal_tiers, max_iter=100)
    graph = ges.fit(data)
    
    print(f"\nLearned edges:")
    for edge in sorted(graph.edges):
        print(f"  {edge[0]} -> {edge[1]}")
    
    # Check recovery
    edges_set = set(graph.edges)
    expected_edges = {("X1", "X3"), ("X2", "X3"), ("X3", "Y"), ("X1", "Y")}
    
    print(f"\n✓ Expected edges: {expected_edges}")
    print(f"✓ Learned edges: {edges_set}")
    
    recovered = expected_edges & edges_set
    print(f"\n✓ Correctly recovered: {len(recovered)}/{len(expected_edges)} edges")
    for edge in recovered:
        print(f"    {edge[0]} -> {edge[1]}")
    
    return graph


if __name__ == "__main__":
    print("Testing Causal Discovery Algorithms on Synthetic Data\n")
    
    print("True Causal Structure:")
    print("  X1 -> X3 -> Y")
    print("  X2 -> X3")
    print("  X1 -> Y")
    print()
    
    pc_graph = test_pc_algorithm()
    ges_graph = test_ges_algorithm()
    
    print("\n" + "=" * 60)
    print("✓ Tests completed successfully!")
    print("=" * 60)
