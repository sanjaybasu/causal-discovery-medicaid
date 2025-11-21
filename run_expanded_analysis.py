"""Run expanded causal discovery with detailed intervention types on larger sample."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from causal_discovery.data_loader_enhanced import (
    load_causal_dataset_enhanced_optimized,
   TemporalConfig,
)
from causal_discovery.algorithms import (
    PCAlgorithm,
    GESAlgorithm,
    CausalGraph,
)

# Configuration
RESULTS_DIR = REPO_ROOT / "results" / "causal_discovery_expanded"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
plt.rcParams["figure.figsize"] = (18, 12)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 9


def visualize_causal_graph(
    graph: CausalGraph,
    temporal_tiers: list,
    title: str = "Causal Graph",
    output_path: Path = None,
):
    """Create publication-quality visualization."""
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)
    
    # Assign colors
    node_colors = []
    tier_labels = {0: "Baseline", 1: "Treatment", 2: "Outcomes"}
    node_tier = {}
    
    for node in G.nodes():
        for tier_idx, tier in enumerate(temporal_tiers):
            if node in tier:
                node_tier[node] = tier_idx
                if tier_idx == 0:
                    node_colors.append("#6BAED6")
                elif tier_idx == 1:
                    node_colors.append("#74C476")
                elif tier_idx == 2:
                    node_colors.append("#FD8D3C")
                break
        else:
            node_colors.append("#CCCCCC")
            node_tier[node] = -1
    
    # Layout
    pos = {}
    tier_positions = {0: [], 1: [], 2: []}
    
    for node in G.nodes():
        tier = node_tier.get(node, -1)
        if tier >= 0:
            tier_positions[tier].append(node)
    
    for tier_idx in range(3):
        nodes_in_tier = tier_positions[tier_idx]
        n_nodes = len(nodes_in_tier)
        for i, node in enumerate(nodes_in_tier):
            x = tier_idx * 5
            y = (i - n_nodes / 2) * 1.5
            pos[node] = (x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=3000, alpha=0.9, ax=ax,
        edgecolors="black", linewidths=2
    )
    
    # Labels
    labels = {node: node.replace("_", "\n") for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=14, font_weight="bold", ax=ax
    )
    
    # Edges
    nx.draw_networkx_edges(
        G, pos, edge_color="black",
        arrows=True, arrowsize=20, width=2,
        ax=ax, arrowstyle="-|>", connectionstyle="arc3,rad=0.1"
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6BAED6", edgecolor="black", label="Baseline"),
        Patch(facecolor="#74C476", edgecolor="black", label="Treatment"),
        Patch(facecolor="#FD8D3C", edgecolor="black", label="Outcomes"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved graph to {output_path}")
    
    plt.close()


def analyze_mechanisms(graph: CausalGraph, algorithm_name: str) -> dict:
    """Analyze causal mechanisms."""
    adj = graph.to_adjacency_dict()
    
    results = {
        "algorithm": algorithm_name,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "intervention_effects": [],
        "baseline_predictors": {},
        "intervention_drivers": [],
    }
    
    # Treatment effects (focus on specialty-specific interventions)
    treatment_nodes = [n for n in graph.nodes if any(
        keyword in n for keyword in ["therapy", "pharmacy", "chw", "care_coord", "phone", "sms"]
    )]
    
    for treatment in treatment_nodes:
        children = adj[treatment]["children"]
        for child in children:
            # Only log effects on outcomes
            if any(keyword in child for keyword in ["followup", "outcome"]):
                results["intervention_effects"].append({
                    "treatment": treatment,
                    "outcome": child,
                })
    
    # Baseline predictors
    outcome_nodes = [n for n in graph.nodes if "followup" in n]
    for outcome in outcome_nodes:
        parents = adj[outcome]["parents"]
        baseline_parents = [
            p for p in parents
            if "baseline" in p or p in ["age", "risk_score", "gender_female", "race"]
        ]
        if baseline_parents:
            results["baseline_predictors"][outcome] = baseline_parents
    
    # Intervention drivers
    for treatment in treatment_nodes:
        parents = adj[treatment]["parents"]
        for parent in parents:
            if "baseline" in parent or parent in ["age", "risk_score", "gender_female"]:
                results["intervention_drivers"].append({
                    "driver": parent,
                    "treatment": treatment,
                })
    
    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("EXPANDED CAUSAL DISCOVERY WITH DETAILED INTERVENTIONS")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading data with enhanced intervention features...")
    config = TemporalConfig(
        baseline_months=6,
        followup_months=6,
        intervention_buffer_days=30,
    )
    
    # Use full dataset
    dataset, metadata = load_causal_dataset_enhanced_optimized(
        config=config,
        sample_size=None,  # Use full dataset
    )
    
    print(f"   Dataset shape: {dataset.shape}")
    print(f"   Members: {metadata['n_members']}")
    
    # 2. Define variables
    print("\n2. Defining variables with detailed intervention types...")
    
    baseline_vars = [
        "age",
        "gender_female",
        "risk_score",
        "baseline_ed_ct",
        "baseline_ip_ct",
        "baseline_total_paid",
    ]
    
    # Detailed intervention variables by specialty
    treatment_vars = [
        "therapy_any",
        "therapy_count",
        "pharmacy_any",
        "pharmacy_count",
        "chw_any",
        "chw_count",
        "care_coord_any",
        "care_coord_count",
        "phone_any",
        "sms_any",
    ]
    
    outcome_vars = [
        "followup_ed_ct",
        "followup_ip_ct",
        "followup_total_paid",
    ]
    
    all_vars = baseline_vars + treatment_vars + outcome_vars
    selected_vars = [v for v in all_vars if v in dataset.columns]
    
    temporal_tiers = [
        [v for v in baseline_vars if v in selected_vars],
        [v for v in treatment_vars if v in selected_vars],
        [v for v in outcome_vars if v in selected_vars],
    ]
    
    print(f"   Selected {len(selected_vars)} variables:")
    print(f"     Tier 0 (Baseline): {len(temporal_tiers[0])} vars")
    print(f"     Tier 1 (Treatment): {len(temporal_tiers[1])} vars - {temporal_tiers[1]}")
    print(f"     Tier 2 (Outcomes): {len(temporal_tiers[2])} vars")
    
    # Prepare analysis dataset
    analysis_data = dataset[selected_vars].copy()
    analysis_data = analysis_data.dropna()
    
    print(f"   Analysis dataset: {analysis_data.shape} (after dropna)")
    
    # Print intervention exposure statistics
    print("\n   Intervention Exposure Statistics:")
    if "therapy_any" in analysis_data.columns:
        print(f"     Therapy: {analysis_data['therapy_any'].sum()} ({100*analysis_data['therapy_any'].mean():.1f}%)")
    if "pharmacy_any" in analysis_data.columns:
        print(f"     Pharmacy: {analysis_data['pharmacy_any'].sum()} ({100*analysis_data['pharmacy_any'].mean():.1f}%)")
    if "chw_any" in analysis_data.columns:
        print(f"     CHW: {analysis_data['chw_any'].sum()} ({100*analysis_data['chw_any'].mean():.1f}%)")
    if "care_coord_any" in analysis_data.columns:
        print(f"     Care Coord: {analysis_data['care_coord_any'].sum()} ({100*analysis_data['care_coord_any'].mean():.1f}%)")
    
    # 3. Run PC Algorithm
    print("\n3. Running PC Algorithm...")
    pc = PCAlgorithm(
        alpha=0.05,
        max_conditioning_set_size=3,
        temporal_tiers=temporal_tiers,
    )
    pc_graph = pc.fit(analysis_data, variable_names=selected_vars)
    
    print(f"   PC Results: {len(pc_graph.edges)} directed edges, {len(pc_graph.undirected_edges)} undirected")
    
    # 4. Run GES Algorithm
    print("\n4. Running GES Algorithm...")
    ges = GESAlgorithm(
        temporal_tiers=temporal_tiers,
        max_iter=100,
    )
    ges_graph = ges.fit(analysis_data, variable_names=selected_vars)
    
    print(f"   GES Results: {len(ges_graph.edges)} directed edges")
    
    # 5. Analyze mechanisms
    print("\n5. Analyzing intervention-specific mechanisms...")
    pc_mechanisms = analyze_mechanisms(pc_graph, "PC")
    ges_mechanisms = analyze_mechanisms(ges_graph, "GES")
    
    print(f"\n   PC Intervention Effects: {len(pc_mechanisms['intervention_effects'])}")
    for effect in sorted(pc_mechanisms["intervention_effects"], key=lambda x: x['treatment']):
        print(f"     {effect['treatment']} → {effect['outcome']}")
    
    print(f"\n   GES Intervention Effects: {len(ges_mechanisms['intervention_effects'])}")
    for effect in sorted(ges_mechanisms["intervention_effects"], key=lambda x: x['treatment']):
        print(f"     {effect['treatment']} → {effect['outcome']}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    
    visualize_causal_graph(
        pc_graph,
        temporal_tiers,
        title="PC Algorithm: Expanded Causal Graph with Detailed Interventions",
        output_path=RESULTS_DIR / "pc_graph_expanded.png",
    )
    
    visualize_causal_graph(
        ges_graph,
        temporal_tiers,
        title="GES Algorithm: Expanded Causal Graph with Detailed Interventions",
        output_path=RESULTS_DIR / "ges_graph_expanded.png",
    )
    
    # 7. Export results
    print("\n7. Exporting results...")
    
    pd.DataFrame(pc_graph.edges, columns=["from", "to"]).to_csv(
        RESULTS_DIR / "pc_edges_expanded.csv", index=False
    )
    pd.DataFrame(ges_graph.edges, columns=["from", "to"]).to_csv(
        RESULTS_DIR / "ges_edges_expanded.csv", index=False
    )
    
    with open(RESULTS_DIR / "pc_mechanisms_expanded.json", "w") as f:
        json.dump(pc_mechanisms, f, indent=2)
    
    with open(RESULTS_DIR / "ges_mechanisms_expanded.json", "w") as f:
        json.dump(ges_mechanisms, f, indent=2)
    
    # Summary
    summary = {
        "dataset": {
            "n_members": int(metadata["n_members"]),
            "n_variables": len(selected_vars),
            "n_treatment_vars": len(temporal_tiers[1]),
        },
        "pc_results": {
            "n_edges": len(pc_graph.edges),
            "intervention_effects": len(pc_mechanisms["intervention_effects"]),
        },
        "ges_results": {
            "n_edges": len(ges_graph.edges),
            "intervention_effects": len(ges_mechanisms["intervention_effects"]),
        },
    }
    
    with open(RESULTS_DIR / "summary_expanded.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to {RESULTS_DIR}")
    print("\n" + "=" * 80)
    print("EXPANDED ANALYSIS COMPLETE")
    print("=" * 80)
    
    return pc_graph, ges_graph, pc_mechanisms, ges_mechanisms


if __name__ == "__main__":
    main()
