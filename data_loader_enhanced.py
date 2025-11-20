"""Optimized enhanced data loader using vectorized operations."""

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
    
    baseline_months: int = 6
    followup_months: int = 6
    intervention_buffer_days: int = 30


def _ensure_datetime(series: pd.Series) -> pd.Series:
    """Convert series to datetime without timezone."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def load_causal_dataset_enhanced_optimized(
    activation_path: Optional[Path] = None,
    data_root: Path = DEFAULT_DATA_ROOT,
    config: Optional[TemporalConfig] = None,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Optimized version using vectorized operations and parquet files.
    
    This version processes data much faster by:
    1. Using parquet files where available
    2. Filtering data early to reduce memory
    3. Using vectorized pandas operations instead of row-by-row iteration
    """
    config = config or TemporalConfig()
    
    if activation_path is None:
        activation_path = (
            REPO_ROOT / "notebooks" / "engagement_analysis" / "v5 - 20251028" /
            "rr_causal_outputs" / "rr_activation_member_level.parquet"
        )
    
    print(f"Loading activation data from {activation_path}...")
    if activation_path.suffix == ".parquet":
        activation_df = pd.read_parquet(activation_path)
    else:
        activation_df = pd.read_csv(activation_path)
    
    # Sample if requested
    if sample_size and len(activation_df) > sample_size:
        activation_df = activation_df.sample(n=sample_size, random_state=42)
    
    activation_df["activation_ts"] = _ensure_datetime(activation_df["activation_ts"])
    member_ids = set(activation_df["member_id"].unique())
    
    print(f"Processing {len(activation_df)} members...")
    
    # Load outcomes (use parquet for speed)
    print("Loading outcomes...")
    outcomes_path = data_root / "outcomes_monthly.parquet"
    if not outcomes_path.exists():
        outcomes_path = data_root / "outcomes_monthly.csv"
    outcomes = pd.read_parquet(outcomes_path) if outcomes_path.suffix == ".parquet" else pd.read_csv(outcomes_path)
    outcomes["month_year"] = pd.to_datetime(outcomes["month_year"])
    
    # Filter outcomes to only relevant members
    outcomes = outcomes[outcomes["member_id"].isin(member_ids)].copy()
    
    # Load member attributes
    print("Loading member attributes...")
    attrs_path = data_root / "member_attributes.parquet"
    if not attrs_path.exists():
        attrs_path = data_root / "member_attributes.csv"
    attributes = pd.read_parquet(attrs_path) if attrs_path.suffix == ".parquet" else pd.read_csv(attrs_path)
    if "birth_date" in attributes.columns:
        attributes["birth_date"] = _ensure_datetime(attributes["birth_date"])
    attributes = attributes[attributes["member_id"].isin(member_ids)].copy()
    
    # Load encounters
    print("Loading encounters...")
    enc_path = data_root / "encounters.parquet"
    if not enc_path.exists():
        enc_path = data_root / "encounters.csv"
    encounters = pd.read_parquet(enc_path) if enc_path.suffix == ".parquet" else pd.read_csv(enc_path)
    encounters["created_at"] = _ensure_datetime(encounters["created_at"])
    
    # Load waymarker specialty
    print("Loading team member specialties...")
    spec_path = data_root / "waymarker_specialty.parquet"
    if not spec_path.exists():
        spec_path = data_root / "waymarker_specialty.csv"
    waymarker_spec = pd.read_parquet(spec_path) if spec_path.suffix == ".parquet" else pd.read_csv(spec_path)
    
    # Merge encounter with specialty
    print("Merging encounters with specialties...")
    encounters = encounters.merge(
        waymarker_spec[["waymarker_id", "discipline"]],
        left_on="created_by_waymarker_id",
        right_on="waymarker_id",
        how="left"
    )
    
    # Categorize encounters by discipline
    encounters["is_therapy"] = encounters["discipline"].str.contains("Therapist", case=False, na=False).astype(int)
    encounters["is_pharmacy"] = encounters["discipline"].str.contains("Pharmac", case=False, na=False).astype(int)
    encounters["is_chw"] = encounters["discipline"].str.contains("Community Health Worker|CHW", case=False, na=False).astype(int)
    encounters["is_care_coord"] = encounters["discipline"].str.contains("Care Coordinator", case=False, na=False).astype(int)
    encounters["is_phone"] = (encounters["contact_type"] == "PHONE_CALL").astype(int)
    encounters["is_sms"] = (encounters["contact_type"] == "SMS_TEXT").astype(int)
    
    # Now process outcomes and interventions vectorized by member
    print("Processing baseline and follow-up periods...")
    
    # Merge activation dates
    outcomes = outcomes.merge(
        activation_df[["member_id", "activation_ts"]],
        on="member_id",
        how="inner"
    )
    
    encounters = encounters.merge(
        activation_df[["member_id", "activation_ts"]].rename(columns={"member_id": "patient_id"}),
        on="patient_id",
        how="inner"
    )
    
    # Define periods
    outcomes["baseline_start"] = outcomes["activation_ts"] - pd.DateOffset(months=config.baseline_months)
    outcomes["baseline_end"] = outcomes["activation_ts"] - pd.Timedelta(days=1)
    outcomes["followup_start"] = outcomes["activation_ts"] + pd.Timedelta(days=config.intervention_buffer_days)
    outcomes["followup_end"] = outcomes["activation_ts"] + pd.DateOffset(months=config.followup_months)
    
    encounters["followup_start"] = encounters["activation_ts"] + pd.Timedelta(days=config.intervention_buffer_days)
    encounters["followup_end"] = encounters["activation_ts"] + pd.DateOffset(months=config.followup_months)
    
    # Filter baseline outcomes
    baseline_outcomes = outcomes[
        (outcomes["month_year"] >= outcomes["baseline_start"]) &
        (outcomes["month_year"] <= outcomes["baseline_end"])
    ].groupby("member_id").agg({
        "emergency_department_ct": "sum",
        "acute_inpatient_ct": "sum",
        "total_paid": "sum",
        "medical_paid": "sum",
        "pharmacy_paid": "sum",
    }).rename(columns={
        "emergency_department_ct": "baseline_ed_ct",
        "acute_inpatient_ct": "baseline_ip_ct",
        "total_paid": "baseline_total_paid",
        "medical_paid": "baseline_medical_paid",
        "pharmacy_paid": "baseline_pharmacy_paid",
    }).reset_index()
    
    # Filter follow-up outcomes
    followup_outcomes = outcomes[
        (outcomes["month_year"] >= outcomes["followup_start"]) &
        (outcomes["month_year"] <= outcomes["followup_end"])
    ].groupby("member_id").agg({
        "emergency_department_ct": "sum",
        "acute_inpatient_ct": "sum",
        "total_paid": "sum",
        "medical_paid": "sum",
        "pharmacy_paid": "sum",
    }).rename(columns={
        "emergency_department_ct": "followup_ed_ct",
        "acute_inpatient_ct": "followup_ip_ct",
        "total_paid": "followup_total_paid",
        "medical_paid": "followup_medical_paid",
        "pharmacy_paid": "followup_pharmacy_paid",
    }).reset_index()
    
    # Filter follow-up interventions
    followup_encounters = encounters[
        (encounters["created_at"] >= encounters["followup_start"]) &
        (encounters["created_at"] <= encounters["followup_end"])
    ]
    
    interventions = followup_encounters.groupby("patient_id").agg({
        "is_therapy": ["sum", lambda x: (x > 0).astype(int).max()],
        "is_pharmacy": ["sum", lambda x: (x > 0).astype(int).max()],
        "is_chw": ["sum", lambda x: (x > 0).astype(int).max()],
        "is_care_coord": ["sum", lambda x: (x > 0).astype(int).max()],
        "is_phone": ["sum", lambda x: (x > 0).astype(int).max()],
        "is_sms": ["sum", lambda x: (x > 0).astype(int).max()],
    })
    
    interventions.columns = [
        "therapy_count", "therapy_any",
        "pharmacy_count", "pharmacy_any",
        "chw_count", "chw_any",
        "care_coord_count", "care_coord_any",
        "phone_count", "phone_any",
        "sms_count", "sms_any",
    ]
    interventions = interventions.reset_index().rename(columns={"patient_id": "member_id"})
    
    # Merge everything
    print("Merging all features...")
    dataset = activation_df[["member_id", "activation_ts"]].copy()
    dataset = dataset.merge(attributes[["member_id", "birth_date", "gender", "risk_score"]], on="member_id", how="left")
    dataset = dataset.merge(baseline_outcomes, on="member_id", how="left")
    dataset = dataset.merge(followup_outcomes, on="member_id", how="left")
    dataset = dataset.merge(interventions, on="member_id", how="left")
    
    # Compute age
    dataset["age"] = (dataset["activation_ts"] - dataset["birth_date"]).dt.days / 365.25
    
    # Encode gender
    dataset["gender_female"] = (dataset["gender"].str.upper() == "F").astype(int)
    
    # Drop unnecessary columns
    dataset = dataset.drop(columns=["activation_ts", "birth_date", "gender"], errors="ignore")
    
    # Fill missing values
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    dataset[numeric_cols] = dataset[numeric_cols].fillna(0)
    
    print(f"Final dataset shape: {dataset.shape}")
    
    metadata = {
        "n_members": len(dataset),
        "config": config,
        "activation_path": str(activation_path),
    }
    
    return dataset, metadata
