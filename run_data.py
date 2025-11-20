"""Run causal discovery on real Waymark data and generate publication-ready outputs."""

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

from causal_discovery.data_loader import (
    load_causal_dataset,
    TemporalConfig,
)
from causal_discovery.algorithms import (
    PCAlgorithm,
    GESAlgorithm,
    CausalGraph,
)

# Configuration
RESULTS_DIR = REPO_ROOT / "results" / "causal_discovery_real"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"


def visualize_causal_graph(
    graph: CausalGraph,
    temporal_tiers: list,
    title: str = "Causal Graph",
    output_path: Path = None,
):
    """Create publication-quality visualization of causal graph."""
    G = nx.DiGraph()
    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)
    
    # Assign colors based on temporal tiers
    node_colors = []
    tier_labels = {
        0: "Baseline/Demographics",
        1: "Treatment",
        2: "Outcomes"
    }
    node_tier = {}
    
    for node in G.nodes():
        for tier_idx, tier in enumerate(temporal_tiers):
            if node in tier:
                node_tier[node] = tier_idx
                if tier_idx == 0:
                    node_colors.append("#6BAED6")  # Blue
                elif tier_idx == 1:
                    node_colors.append("#74C476")  # Green
                elif tier_idx == 2:
                    node_colors.append("#FD8D3C")  # Orange
                break
        else:
            node_colors.append("#CCCCCC")  # Gray
            node_tier[node] = -1
    
    # Layout with hierarchical structure
    pos = {}
    tier_positions = {0: [], 1: [], 2: []}
    
    for node in G.nodes():
        tier = node_tier.get(node, -1)
        if tier >= 0:
            tier_positions[tier].append(node)
    
    # Position nodes in tiers
    for tier_idx in range(3):
        nodes_in_tier = tier_positions[tier_idx]
        n_nodes = len(nodes_in_tier)
        for i, node in enumerate(nodes_in_tier):
            x = tier_idx * 4
            y = (i - n_nodes / 2) * 2
            pos[node] = (x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=4000, alpha=0.9, ax=ax,
        edgecolors="black", linewidths=2
    )
    
    # Draw labels with better formatting
    labels = {node: node.replace("_", "\n") for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=9, font_weight="bold", ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos, edge_color="black",
        arrows=True, arrowsize=25, width=2.5,
        ax=ax, arrowstyle="-|>", connectionstyle="arc3,rad=0.1"
    )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6BAED6", edgecolor="black", label="Baseline/Demographics"),
        Patch(facecolor="#74C476", edgecolor="black", label="Treatment"),
        Patch(facecolor="#FD8D3C", edgecolor="black", label="Outcomes"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12, framealpha=0.9)
    
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved graph to {output_path}")
    
    plt.close()


def analyze_mechanisms(graph: CausalGraph, algorithm_name: str) -> dict:
    """Analyze causal mechanisms and return structured results."""
    adj = graph.to_adjacency_dict()
    
    results = {
        "algorithm": algorithm_name,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "intervention_effects": [],
        "baseline_predictors": {},
        "intervention_drivers": [],
        "mediating_pathways": [],
    }
    
    # Treatment effects
    treatment_nodes = [n for n in graph.nodes if "intervention" in n]
    for treatment in treatment_nodes:
        children = adj[treatment]["children"]
        for child in children:
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
            if "baseline" in p or p in ["age", "risk_score", "gender_female"]
        ]
        if baseline_parents:
            results["baseline_predictors"][outcome] = baseline_parents
    
    # Intervention drivers
    for treatment in treatment_nodes:
        parents = adj[treatment]["parents"]
        for parent in parents:
            results["intervention_drivers"].append({
                "driver": parent,
                "treatment": treatment,
            })
    
    # Mediating pathways
    for treatment in treatment_nodes:
        for mediator in graph.nodes:
            if mediator in treatment_nodes or mediator in outcome_nodes:
                continue
            if mediator in adj[treatment]["children"]:
                for outcome in outcome_nodes:
                    if outcome in adj[mediator]["children"]:
                        results["mediating_pathways"].append({
                            "treatment": treatment,
                            "mediator": mediator,
                            "outcome": outcome,
                        })
    
    return results


def main():
    """Main execution function."""
    print("=" * 80)
    print("CAUSAL DISCOVERY ON REAL WAYMARK DATA")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading data...")
    config = TemporalConfig(
        baseline_months=6,
        followup_months=6,
        intervention_buffer_days=30,
    )
    
    # Sample for computational efficiency
    dataset, metadata = load_causal_dataset(
        config=config,
        sample_size=2000,  # Use 2000 members for better power
    )
    
    print(f"   Dataset shape: {dataset.shape}")
    print(f"   Members: {metadata['n_members']}")
    
    # 2. Define variables and temporal tiers
    print("\n2. Defining variables and temporal structure...")
    
    baseline_vars = [
        "age",
        "gender_female",
        "risk_score",
        "baseline_ed_ct",
        "baseline_ip_ct",
        "baseline_total_paid",
    ]
    
    treatment_vars = [
        "intervention_any",
        "intervention_count",
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
    for i, tier in enumerate(temporal_tiers):
        print(f"     Tier {i}: {tier}")
    
    # Prepare analysis dataset
    analysis_data = dataset[selected_vars].copy()
    analysis_data = analysis_data.dropna()
    
    print(f"   Analysis dataset: {analysis_data.shape} (after dropna)")
    
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
    print("\n5. Analyzing mechanisms...")
    pc_mechanisms = analyze_mechanisms(pc_graph, "PC")
    ges_mechanisms = analyze_mechanisms(ges_graph, "GES")
    
    print(f"\n   PC Intervention Effects: {len(pc_mechanisms['intervention_effects'])}")
    for effect in pc_mechanisms["intervention_effects"]:
        print(f"     {effect['treatment']} → {effect['outcome']}")
    
    print(f"\n   GES Intervention Effects: {len(ges_mechanisms['intervention_effects'])}")
    for effect in ges_mechanisms["intervention_effects"]:
        print(f"     {effect['treatment']} → {effect['outcome']}")
    
    print(f"\n   PC Mediating Pathways: {len(pc_mechanisms['mediating_pathways'])}")
    for pathway in pc_mechanisms["mediating_pathways"]:
        print(f"     {pathway['treatment']} → {pathway['mediator']} → {pathway['outcome']}")
    
    print(f"\n   GES Mediating Pathways: {len(ges_mechanisms['mediating_pathways'])}")
    for pathway in ges_mechanisms["mediating_pathways"]:
        print(f"     {pathway['treatment']} → {pathway['mediator']} → {pathway['outcome']}")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    
    visualize_causal_graph(
        pc_graph,
        temporal_tiers,
        title="PC Algorithm: Learned Causal Graph from Waymark Data",
        output_path=RESULTS_DIR / "pc_graph.png",
    )
    
    visualize_causal_graph(
        ges_graph,
        temporal_tiers,
        title="GES Algorithm: Learned Causal Graph from Waymark Data",
        output_path=RESULTS_DIR / "ges_graph.png",
    )
    
    # 7. Export results
    print("\n7. Exporting results...")
    
    # Save edge lists
    pd.DataFrame(pc_graph.edges, columns=["from", "to"]).to_csv(
        RESULTS_DIR / "pc_edges.csv", index=False
    )
    pd.DataFrame(ges_graph.edges, columns=["from", "to"]).to_csv(
        RESULTS_DIR / "ges_edges.csv", index=False
    )
    
    # Save mechanism analysis
    with open(RESULTS_DIR / "pc_mechanisms.json", "w") as f:
        json.dump(pc_mechanisms, f, indent=2)
    
    with open(RESULTS_DIR / "ges_mechanisms.json", "w") as f:
        json.dump(ges_mechanisms, f, indent=2)
    
    # Save summary statistics
    summary = {
        "dataset": {
            "n_members": int(metadata["n_members"]),
            "n_variables": len(selected_vars),
            "temporal_config": {
                "baseline_months": config.baseline_months,
                "followup_months": config.followup_months,
            },
        },
        "pc_results": {
            "n_edges": len(pc_graph.edges),
            "n_undirected": len(pc_graph.undirected_edges),
            "intervention_effects": len(pc_mechanisms["intervention_effects"]),
            "mediating_pathways": len(pc_mechanisms["mediating_pathways"]),
        },
        "ges_results": {
            "n_edges": len(ges_graph.edges),
            "intervention_effects": len(ges_mechanisms["intervention_effects"]),
            "mediating_pathways": len(ges_mechanisms["mediating_pathways"]),
        },
    }
    
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ All results saved to {RESULTS_DIR}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return pc_graph, ges_graph, pc_mechanisms, ges_mechanisms


if __name__ == "__main__":
    main()
