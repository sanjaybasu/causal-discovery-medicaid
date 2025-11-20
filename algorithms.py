"""Causal discovery algorithms: PC (constraint-based) and GES (score-based).

This module implements simplified versions of the PC and GES algorithms
for learning causal graphs from observational data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CausalGraph:
    """Represents a learned causal graph."""
    
    nodes: List[str]
    edges: List[Tuple[str, str]]  # Directed edges (from, to)
    undirected_edges: List[Tuple[str, str]]  # Undirected edges for CPDAG
    
    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def to_adjacency_dict(self) -> dict:
        """Convert to adjacency list representation."""
        adj = {node: {"parents": [], "children": []} for node in self.nodes}
        for from_node, to_node in self.edges:
            adj[to_node]["parents"].append(from_node)
            adj[from_node]["children"].append(to_node)
        return adj


def partial_correlation(
    data: np.ndarray,
    i: int,
    j: int,
    conditioning_set: Set[int],
) -> Tuple[float, float]:
    """Compute partial correlation and p-value using Fisher's Z-transform.
    
    Args:
        data: n x p data matrix
        i, j: Indices of variables to test
        conditioning_set: Set of variable indices to condition on
    
    Returns:
        Tuple of (correlation, p-value)
    """
    n = data.shape[0]
    
    if len(conditioning_set) == 0:
        # Marginal correlation
        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
    else:
        # Partial correlation via regression residuals
        cond_indices = list(conditioning_set)
        Z = data[:, cond_indices]
        
        # Add intercept
        Z = np.column_stack([np.ones(n), Z])
        
        # Regress X_i on Z
        beta_i = np.linalg.lstsq(Z, data[:, i], rcond=None)[0]
        resid_i = data[:, i] - Z @ beta_i
        
        # Regress X_j on Z
        beta_j = np.linalg.lstsq(Z, data[:, j], rcond=None)[0]
        resid_j = data[:, j] - Z @ beta_j
        
        # Correlation of residuals
        corr = np.corrcoef(resid_i, resid_j)[0, 1]
    
    # Fisher's Z-transform
    if abs(corr) >= 0.9999:
        corr = 0.9999 * np.sign(corr)
    
    z_score = 0.5 * np.log((1 + corr) / (1 - corr))
    
    # Standard error
    k = len(conditioning_set)
    se = 1.0 / np.sqrt(n - k - 3)
    
    # Test statistic
    z_stat = z_score / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return corr, p_value


class PCAlgorithm:
    """PC (Peter-Clark) algorithm for constraint-based causal discovery.
    
    The PC algorithm learns the Markov equivalence class (CPDAG) by:
    1. Starting with a complete undirected graph
    2. Removing edges based on conditional independence tests
    3. Orienting edges using collider detection and Meek rules
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        max_conditioning_set_size: Optional[int] = 3,
        temporal_tiers: Optional[List[List[str]]] = None,
    ):
        """Initialize PC algorithm.
        
        Args:
            alpha: Significance level for independence tests
            max_conditioning_set_size: Maximum size of conditioning sets
            temporal_tiers: List of variable groups in temporal order (earlier -> later)
        """
        self.alpha = alpha
        self.max_cond_size = max_conditioning_set_size
        self.temporal_tiers = temporal_tiers
        self.graph: Optional[CausalGraph] = None
    
    def _is_temporally_valid(self, from_var: str, to_var: str) -> bool:
        """Check if edge from_var -> to_var respects temporal constraints."""
        if self.temporal_tiers is None:
            return True
        
        # Find tier indices
        from_tier = None
        to_tier = None
        
        for idx, tier in enumerate(self.temporal_tiers):
            if from_var in tier:
                from_tier = idx
            if to_var in tier:
                to_tier = idx
        
        if from_tier is None or to_tier is None:
            return True  # Variable not in tiers, allow
        
        # Only allow from earlier or same tier to later tier
        return from_tier <= to_tier
    
    def fit(self, data: pd.DataFrame, variable_names: Optional[List[str]] = None) -> CausalGraph:
        """Learn causal graph from data.
        
        Args:
            data: DataFrame where each column is a variable
            variable_names: Optional list of variable names to use (default: all columns)
        
        Returns:
            Learned causal graph (CPDAG)
        """
        if variable_names is None:
            variable_names = list(data.columns)
        
        # Convert to numpy array
        data_array = data[variable_names].to_numpy(dtype=float)
        n_vars = len(variable_names)
        
        # Phase 1: Skeleton discovery
        # Start with complete undirected graph
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)
        
        # Store separation sets
        sep_sets = {}
        
        # Iterate through conditioning set sizes
        max_depth = self.max_cond_size if self.max_cond_size else n_vars - 2
        
        for depth in range(max_depth + 1):
            edge_removed = False
            
            for i, j in combinations(range(n_vars), 2):
                if not adjacency[i, j]:
                    continue  # Already removed
                
                # Get neighbors of i (excluding j)
                neighbors_i = set(np.where(adjacency[i, :])[0]) - {j}
                
                # Test all conditioning sets of size 'depth'
                if len(neighbors_i) < depth:
                    continue
                
                for cond_set in combinations(neighbors_i, depth):
                    cond_set = set(cond_set)
                    
                    # Test conditional independence
                    _, p_value = partial_correlation(data_array, i, j, cond_set)
                    
                    if p_value > self.alpha:
                        # Independent: remove edge
                        adjacency[i, j] = False
                        adjacency[j, i] = False
                        sep_sets[(i, j)] = cond_set
                        sep_sets[(j, i)] = cond_set
                        edge_removed = True
                        break
            
            if not edge_removed:
                break  # No more edges to remove at this depth
        
        # Phase 2: Edge orientation
        # 2a. Detect v-structures (colliders): i -> k <- j where i and j are not adjacent
        directed_edges = set()
        undirected_edges = set()
        
        for i, j in combinations(range(n_vars), 2):
            if adjacency[i, j]:
                undirected_edges.add((min(i, j), max(i, j)))
        
        # Find v-structures
        for k in range(n_vars):
            neighbors_k = set(np.where(adjacency[k, :])[0])
            
            for i, j in combinations(neighbors_k, 2):
                if adjacency[i, j]:
                    continue  # i and j are adjacent, not a v-structure
                
                # Check if k is NOT in the separation set of i and j
                if (i, j) in sep_sets and k not in sep_sets[(i, j)]:
                    # Orient as i -> k <- j
                    directed_edges.add((i, k))
                    directed_edges.add((j, k))
                    # Remove from undirected
                    undirected_edges.discard((min(i, k), max(i, k)))
                    undirected_edges.discard((min(j, k), max(j, k)))
        
        # 2b. Apply Meek rules (simplified)
        # Rule 1: i -> j - k => i -> j -> k (to avoid new v-structure)
        # Rule 2: i -> j -> k, i - k => i -> k (to avoid cycle)
        changed = True
        max_iterations = 10
        iteration = 0
        
        while changed and iteration < max_iterations:
            changed = False
            iteration += 1
            
            for i, k in list(undirected_edges):
                # Check Rule 1 and Rule 2
                # Rule 1: Find j such that j -> i and j not adjacent to k
                for j in range(n_vars):
                    if (j, i) in directed_edges and not adjacency[j, k]:
                        # Orient i - k as i -> k
                        directed_edges.add((i, k))
                        undirected_edges.discard((min(i, k), max(i, k)))
                        changed = True
                        break
                
                if changed:
                    continue
                
                # Rule 2: Find j such that i -> j -> k
                for j in range(n_vars):
                    if (i, j) in directed_edges and (j, k) in directed_edges:
                        # Orient i - k as i -> k
                        directed_edges.add((i, k))
                        undirected_edges.discard((min(i, k), max(i, k)))
                        changed = True
                        break
        
        # Apply temporal constraints
        if self.temporal_tiers:
            directed_edges_filtered = set()
            for i, j in directed_edges:
                var_i = variable_names[i]
                var_j = variable_names[j]
                if self._is_temporally_valid(var_i, var_j):
                    directed_edges_filtered.add((i, j))
            directed_edges = directed_edges_filtered
        
        # Convert to node names
        edges = [(variable_names[i], variable_names[j]) for i, j in directed_edges]
        undirected = [(variable_names[i], variable_names[j]) for i, j in undirected_edges]
        
        self.graph = CausalGraph(
            nodes=variable_names,
            edges=edges,
            undirected_edges=undirected,
        )
        
        return self.graph


class GESAlgorithm:
    """Greedy Equivalence Search (GES) for score-based causal discovery.
    
    GES is a score-based algorithm that:
    1. Starts with an empty graph
    2. Forward phase: Greedily adds edges that improve the score
    3. Backward phase: Greedily removes edges that improve the score
    """
    
    def __init__(
        self,
        temporal_tiers: Optional[List[List[str]]] = None,
        max_iter: int = 100,
    ):
        """Initialize GES algorithm.
        
        Args:
            temporal_tiers: List of variable groups in temporal order
            max_iter: Maximum iterations for forward/backward phases
        """
        self.temporal_tiers = temporal_tiers
        self.max_iter = max_iter
        self.graph: Optional[CausalGraph] = None
    
    def _compute_bic_score(
        self,
        data: np.ndarray,
        target_idx: int,
        parent_indices: Set[int],
    ) -> float:
        """Compute BIC score for a variable given its parents.
        
        Args:
            data: n x p data matrix
            target_idx: Index of target variable
            parent_indices: Indices of parent variables
        
        Returns:
            BIC score (higher is better)
        """
        n = data.shape[0]
        
        if len(parent_indices) == 0:
            # No parents: use variance of target
            variance = np.var(data[:, target_idx])
            log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * n
            num_params = 1
        else:
            # Regression on parents
            parent_list = list(parent_indices)
            X = data[:, parent_list]
            X = np.column_stack([np.ones(n), X])  # Add intercept
            y = data[:, target_idx]
            
            # Fit linear regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            variance = np.var(residuals)
            
            if variance < 1e-10:
                variance = 1e-10
            
            log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - 0.5 * np.sum(residuals**2) / variance
            num_params = len(parent_list) + 1  # coefficients + intercept
        
        # BIC = log-likelihood - (k/2) * log(n)
        bic = log_likelihood - (num_params / 2) * np.log(n)
        return bic
    
    def _is_temporally_valid(self, from_var: str, to_var: str) -> bool:
        """Check if edge from_var -> to_var respects temporal constraints."""
        if self.temporal_tiers is None:
            return True
        
        from_tier = None
        to_tier = None
        
        for idx, tier in enumerate(self.temporal_tiers):
            if from_var in tier:
                from_tier = idx
            if to_var in tier:
                to_tier = idx
        
        if from_tier is None or to_tier is None:
            return True
        
        return from_tier <= to_tier
    
    def fit(self, data: pd.DataFrame, variable_names: Optional[List[str]] = None) -> CausalGraph:
        """Learn causal graph from data using GES.
        
        Args:
            data: DataFrame where each column is a variable
            variable_names: Optional list of variable names to use
        
        Returns:
            Learned causal graph (DAG)
        """
        if variable_names is None:
            variable_names = list(data.columns)
        
        data_array = data[variable_names].to_numpy(dtype=float)
        n_vars = len(variable_names)
        
        # Initialize with empty graph (no edges)
        parents = {i: set() for i in range(n_vars)}
        
        # Forward phase: Add edges
        for iteration in range(self.max_iter):
            best_score_delta = 0
            best_edge = None
            
            for i, j in combinations(range(n_vars), 2):
                # Try adding i -> j
                if i not in parents[j]:
                    var_i = variable_names[i]
                    var_j = variable_names[j]
                    
                    if not self._is_temporally_valid(var_i, var_j):
                        continue
                    
                    # Compute score before and after
                    score_before = self._compute_bic_score(data_array, j, parents[j])
                    new_parents = parents[j] | {i}
                    score_after = self._compute_bic_score(data_array, j, new_parents)
                    score_delta = score_after - score_before
                    
                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_edge = (i, j)
                
                # Try adding j -> i
                if j not in parents[i]:
                    var_i = variable_names[i]
                    var_j = variable_names[j]
                    
                    if not self._is_temporally_valid(var_j, var_i):
                        continue
                    
                    score_before = self._compute_bic_score(data_array, i, parents[i])
                    new_parents = parents[i] | {j}
                    score_after = self._compute_bic_score(data_array, i, new_parents)
                    score_delta = score_after - score_before
                    
                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_edge = (j, i)
            
            if best_edge is None:
                break  # No improvement
            
            # Add best edge
            from_node, to_node = best_edge
            parents[to_node].add(from_node)
        
        # Backward phase: Remove edges
        for iteration in range(self.max_iter):
            best_score_delta = 0
            best_edge_to_remove = None
            
            for j in range(n_vars):
                for i in parents[j]:
                    # Try removing i -> j
                    score_before = self._compute_bic_score(data_array, j, parents[j])
                    new_parents = parents[j] - {i}
                    score_after = self._compute_bic_score(data_array, j, new_parents)
                    score_delta = score_after - score_before
                    
                    if score_delta > best_score_delta:
                        best_score_delta = score_delta
                        best_edge_to_remove = (i, j)
            
            if best_edge_to_remove is None:
                break  # No improvement
            
            # Remove edge
            from_node, to_node = best_edge_to_remove
            parents[to_node].remove(from_node)
        
        # Convert to edge list
        edges = []
        for to_node, parent_set in parents.items():
            for from_node in parent_set:
                edges.append((variable_names[from_node], variable_names[to_node]))
        
        self.graph = CausalGraph(
            nodes=variable_names,
            edges=edges,
            undirected_edges=[],
        )
        
        return self.graph
