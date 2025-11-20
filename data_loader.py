"""Data loader for causal discovery analysis.

This module prepares the Waymark data with temporal structure (pre/post periods)
to enable causal discovery with temporal precedence constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "real_inputs"


@dataclass
class TemporalConfig:
    """Configuration for temporal alignment of data."""
    
    baseline_months: int = 6  # Months before intervention to aggregate
    followup_months: int = 6  # Months after intervention to aggregate
    intervention_buffer_days: int = 30  # Days around intervention date to exclude


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert series to datetime without timezone."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def load_outcomes_monthly(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """Load monthly outcomes data."""
    path = data_root / "outcomes_monthly.parquet"
    if not path.exists():
        path = data_root / "outcomes_monthly.csv"
    
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["month_year"] = pd.to_datetime(df["month_year"])
    return df


def load_member_attributes(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """Load member demographic and risk attributes."""
    path = data_root / "member_attributes.parquet"
    if not path.exists():
        path = data_root / "member_attributes.csv"
    
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "birth_date" in df.columns:
        df["birth_date"] = _ensure_datetime(df["birth_date"])
    return df


def load_interventions(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """Load interventions data."""
    path = data_root / "interventions.csv"
    df = pd.read_csv(path)
    df["intervention_date"] = _ensure_datetime(df["intervention_date"])
    return df


def aggregate_outcomes_for_period(
    outcomes: pd.DataFrame,
    member_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prefix: str = "",
) -> dict:
    """Aggregate outcome metrics for a specific time period.
    
    Args:
        outcomes: Monthly outcomes dataframe
        member_id: Member identifier
        start_date: Period start date
        end_date: Period end date (inclusive)
        prefix: Prefix for output column names (e.g., "pre_" or "post_")
    
    Returns:
        Dictionary of aggregated metrics
    """
    member_data = outcomes[
        (outcomes["member_id"] == member_id) &
        (outcomes["month_year"] >= start_date) &
        (outcomes["month_year"] <= end_date)
    ]
    
    if len(member_data) == 0:
        return {
            f"{prefix}ed_ct": 0.0,
            f"{prefix}ip_ct": 0.0,
            f"{prefix}total_paid": 0.0,
            f"{prefix}medical_paid": 0.0,
            f"{prefix}pharmacy_paid": 0.0,
            f"{prefix}months_observed": 0,
        }
    
    return {
        f"{prefix}ed_ct": float(member_data["emergency_department_ct"].sum()),
        f"{prefix}ip_ct": float(member_data["acute_inpatient_ct"].sum()),
        f"{prefix}total_paid": float(member_data["total_paid"].sum()),
        f"{prefix}medical_paid": float(member_data["medical_paid"].sum()),
        f"{prefix}pharmacy_paid": float(member_data["pharmacy_paid"].sum()),
        f"{prefix}months_observed": len(member_data),
    }


def compute_intervention_features(
    interventions: pd.DataFrame,
    member_id: str,
    activation_date: pd.Timestamp,
    config: TemporalConfig,
) -> dict:
    """Compute intervention exposure features during the follow-up period.
    
    Args:
        interventions: Interventions dataframe
        member_id: Member identifier (person_key)
        activation_date: Activation/index date
        config: Temporal configuration
    
    Returns:
        Dictionary of intervention features
    """
    followup_start = activation_date + pd.Timedelta(days=config.intervention_buffer_days)
    followup_end = activation_date + pd.DateOffset(months=config.followup_months)
    
    member_interventions = interventions[
        (interventions["person_key"] == member_id) &
        (interventions["intervention_date"] >= followup_start) &
        (interventions["intervention_date"] <= followup_end)
    ]
    
    if len(member_interventions) == 0:
        return {
            "intervention_any": 0,
            "intervention_count": 0,
        }
    
    return {
        "intervention_any": 1,
        "intervention_count": len(member_interventions),
    }


def prepare_causal_dataset(
    activation_df: pd.DataFrame,
    data_root: Path = DEFAULT_DATA_ROOT,
    config: Optional[TemporalConfig] = None,
) -> pd.DataFrame:
    """Prepare dataset with pre/post temporal structure for causal discovery.
    
    This function creates a dataset where each row represents a member with:
    - Baseline (pre) features: demographics, risk, historical utilization
    - Treatment features: intervention exposure during follow-up
    - Outcome (post) features: utilization and costs after intervention
    
    Args:
        activation_df: DataFrame with member_id and activation_ts columns
        data_root: Path to data directory
        config: Temporal configuration
    
    Returns:
        DataFrame ready for causal discovery
    """
    config = config or TemporalConfig()
    
    # Load all datasets
    outcomes = load_outcomes_monthly(data_root)
    attributes = load_member_attributes(data_root)
    interventions = load_interventions(data_root)
    
    # Ensure activation_ts is datetime
    activation_df = activation_df.copy()
    activation_df["activation_ts"] = _ensure_datetime(activation_df["activation_ts"])
    
    rows = []
    
    for _, row in activation_df.iterrows():
        member_id = row["member_id"]
        activation_date = row["activation_ts"]
        
        if pd.isna(activation_date):
            continue
        
        # Define time periods
        baseline_start = activation_date - pd.DateOffset(months=config.baseline_months)
        baseline_end = activation_date - pd.Timedelta(days=1)
        followup_start = activation_date + pd.Timedelta(days=config.intervention_buffer_days)
        followup_end = activation_date + pd.DateOffset(months=config.followup_months)
        
        # Get baseline outcomes
        baseline_metrics = aggregate_outcomes_for_period(
            outcomes, member_id, baseline_start, baseline_end, prefix="baseline_"
        )
        
        # Get follow-up outcomes
        followup_metrics = aggregate_outcomes_for_period(
            outcomes, member_id, followup_start, followup_end, prefix="followup_"
        )
        
        # Get member attributes
        member_attrs = attributes[attributes["member_id"] == member_id]
        if len(member_attrs) == 0:
            continue
        
        attrs = member_attrs.iloc[0]
        
        # Compute age at activation
        age = None
        if pd.notna(attrs.get("birth_date")):
            age = (activation_date - attrs["birth_date"]).days / 365.25
        
        # Get intervention features
        # Note: interventions use person_key, which might differ from member_id
        # For now, we'll try to match by member_id as person_key
        intervention_features = compute_intervention_features(
            interventions, member_id, activation_date, config
        )
        
        # Combine all features
        record = {
            "member_id": member_id,
            "activation_date": activation_date,
            # Demographics (baseline/static)
            "age": age,
            "gender": attrs.get("gender"),
            "race": attrs.get("race"),
            "risk_score": attrs.get("risk_score"),
            # Baseline outcomes
            **baseline_metrics,
            # Treatment
            **intervention_features,
            # Follow-up outcomes
            **followup_metrics,
        }
        
        rows.append(record)
    
    df = pd.DataFrame(rows)
    
    # Clean and encode categorical variables
    if "gender" in df.columns:
        df["gender_female"] = (df["gender"].str.upper() == "F").astype(int)
        df = df.drop(columns=["gender"])
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def load_causal_dataset(
    activation_path: Optional[Path] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    config: Optional[TemporalConfig] = None,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Load and prepare the full causal discovery dataset.
    
    Args:
        activation_path: Path to activation data (parquet or csv)
        data_root: Path to data directory
        config: Temporal configuration
        sample_size: If provided, randomly sample this many records
    
    Returns:
        Tuple of (dataset DataFrame, metadata dict)
    """
    if activation_path is None:
        # Use default from waymark_causal
        activation_path = (
            REPO_ROOT / "notebooks" / "engagement_analysis" / "v5 - 20251028" /
            "rr_causal_outputs" / "rr_activation_member_level.parquet"
        )
    
    if not activation_path.exists():
        raise FileNotFoundError(f"Activation data not found: {activation_path}")
    
    # Load activation data
    if activation_path.suffix == ".parquet":
        activation_df = pd.read_parquet(activation_path)
    else:
        activation_df = pd.read_csv(activation_path)
    
    # Sample if requested
    if sample_size and len(activation_df) > sample_size:
        activation_df = activation_df.sample(n=sample_size, random_state=42)
    
    # Prepare dataset
    dataset = prepare_causal_dataset(activation_df, data_root, config)
    
    metadata = {
        "n_members": len(dataset),
        "config": config or TemporalConfig(),
        "activation_path": str(activation_path),
    }
    
    return dataset, metadata
