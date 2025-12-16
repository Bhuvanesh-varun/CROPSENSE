import os
import pandas as pd
import numpy as np

# ============================================================
# CONSTANTS
# ============================================================

NUTRIENTS = ["N", "P", "K", "S", "Zn", "Fe", "B", "Mn", "Cu"]


# ============================================================
# SQI / PHI CLASSIFIERS
# ============================================================

def classify_sqi(sqi: float) -> str:
    if sqi <= 1.4:
        return "Very Poor"
    elif sqi <= 2.4:
        return "Poor"
    elif sqi <= 3.4:
        return "Moderate"
    elif sqi <= 4.2:
        return "Good"
    return "Excellent"


def classify_phi(phi: float) -> str:
    if phi <= 3.0:
        return "Severe Stress"
    elif phi <= 5.0:
        return "Moderate Stress"
    elif phi <= 7.5:
        return "At Risk but Recoverable"
    elif phi <= 9.0:
        return "Healthy"
    return "Very Healthy"


# ============================================================
# GROWTH STAGE RELEVANCE WEIGHTS
# ============================================================

# keys are lowercase to match normalized growth stage values
STAGE_RELEVANCE = {
    "germination/seedling": {
        "high": 1.0, "early": 1.0, "mid": 0.7, "medium": 0.7, "low": 0.5, "pH correction": 1.0, "salinity flush": 1.0,
    },
    "vegetative": {
        "high": 1.0, "early": 0.9, "mid": 0.8, "medium": 0.8, "low": 0.6, "pH correction": 1.0, "salinity flush": 1.0,
    },
    "flowering": {
        "high": 1.0, "early": 0.6, "mid": 1.0, "medium": 0.8, "low": 0.6, "pH correction": 1.0, "salinity flush": 1.0,
    },
    "fruiting/grain fill": {
        "high": 1.0, "early": 0.4, "mid": 1.0, "medium": 0.8, "low": 0.6, "pH correction": 1.0, "salinity flush": 1.0,
    },
    "maturity": {
        "high": 0.8, "early": 0.2, "mid": 0.6, "medium": 0.6, "low": 0.5, "pH correction": 1.0, "salinity flush": 1.0,
    },
}


# ============================================================
# LOAD REFERENCE CSVs
# ============================================================

def load_reference_tables(base_path="data"):
    """
    Loads CSV reference tables. Returns 4 DataFrames in order:
    nutrient_thresholds, crop_req, treatment_recs, pest_actions
    If a file is missing, returns an empty DataFrame for that table.
    """
    def _safe_read(path):
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            # Return empty dataframe if missing; caller must handle empties.
            return pd.DataFrame()

    nutrient_thresholds = _safe_read(os.path.join(base_path, "nutrient_thresholds.csv"))
    crop_req = _safe_read(os.path.join(base_path, "crop_nutrient_requirement.csv"))
    treatment_recs = _safe_read(os.path.join(base_path, "treatment_recommendations.csv"))
    pest_actions = _safe_read(os.path.join(base_path, "pest_disease_control.csv"))

    # Normalize some column names to help later lookups (if present)
    if not nutrient_thresholds.empty:
        # ensure expected columns exist; if not, leave as-is (caller must handle)
        nutrient_thresholds.columns = [c.strip() for c in nutrient_thresholds.columns]

    return nutrient_thresholds, crop_req, treatment_recs, pest_actions


# ============================================================
# THRESHOLD LOOKUP
# ============================================================

def _get_threshold_row(nutrient_thresholds: pd.DataFrame, crop_name: str, nutrient: str):
    """
    Returns a Series for the nutrient threshold for crop_name and nutrient.
    Falls back to 'generic' crop row if crop-specific not found.
    Case-insensitive matching; returns None if nothing found.
    """
    if nutrient_thresholds.empty:
        return None

    # normalize string comparisons
    crop_name_l = "" if crop_name is None else str(crop_name).strip().lower()
    nutrient_l = "" if nutrient is None else str(nutrient).strip().lower()

    # Make a normalized copy of relevant columns for safe matching
    nt = nutrient_thresholds.copy()
    nt_cols = list(nt.columns)
    # try to ensure these columns exist
    expected_cols = ["Crop_Name", "Nutrient"]
    if not all(col in nt_cols for col in expected_cols):
        return None

    # lowercase helpers
    crop_series = nt["Crop_Name"].astype(str).str.strip().str.lower()
    nutrient_series = nt["Nutrient"].astype(str).str.strip().str.lower()

    mask = (crop_series == crop_name_l) & (nutrient_series == nutrient_l)
    sel = nt[mask]
    if sel.empty:
        # fallback to generic
        mask2 = (crop_series == "generic") & (nutrient_series == nutrient_l)
        sel = nt[mask2]

    if sel.empty:
        return None

    # return first matching row as Series
    return sel.iloc[0]


# ============================================================
# NUTRIENT STATUS ENGINE
# ============================================================

def assess_nutrient_status(row: dict, nutrient_thresholds: pd.DataFrame):
    """
    For each nutrient in NUTRIENTS, compare row value to threshold and
    return list of dicts: nutrient, value, status, severity_score
    """
    results = []

    nutrient_to_col = {
        "N": "Available_N_Kg_Ha", "P": "Available_P_Kg_Ha", "K": "Available_K_Kg_Ha",
        "S": "Available_S_Kg_Ha", "Zn": "Available_Zn_Ppm", "B": "Available_B_Ppm",
        "Fe": "Available_Fe_Ppm", "Mn": "Available_Mn_Ppm", "Cu": "Available_Cu_Ppm",
    }

    crop_name = row.get("Crop_Name", "")

    for nut in NUTRIENTS:
        col = nutrient_to_col.get(nut)
        # if expected column not present in row, mark Unknown
        raw_value = row.get(col, np.nan)
        try:
            value = float(raw_value) if (raw_value is not None and not (isinstance(raw_value, float) and np.isnan(raw_value))) else np.nan
        except Exception:
            value = np.nan

        thr = _get_threshold_row(nutrient_thresholds, crop_name, nut)
        if thr is None or np.isnan(value):
            results.append({"nutrient": nut, "value": value, "status": "Unknown", "severity_score": 0.0})
            continue

        # Safely read threshold columns (use get with fallback)
        # Accept both numeric and string columns
        def _safe_float(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        low = _safe_float(thr.get("Low_Critical") if "Low_Critical" in thr else thr.get("Low", np.nan))
        opt_min = _safe_float(thr.get("Optimal_Min") if "Optimal_Min" in thr else thr.get("Opt_Min", np.nan))
        opt_max = _safe_float(thr.get("Optimal_Max") if "Optimal_Max" in thr else thr.get("Opt_Max", np.nan))
        high = _safe_float(thr.get("High_Critical") if "High_Critical" in thr else thr.get("High", np.nan))

        # If thresholds not provided, return Unknown
        if any(np.isnan(x) for x in [low, opt_min, opt_max, high]):
            results.append({"nutrient": nut, "value": value, "status": "Unknown", "severity_score": 0.0})
            continue

        if value < low:
            # severity scaled 0..10 proportional to shortfall (clamped)
            sev = min(10.0, (low - value) / max(low, 1e-6) * 10)
            status = "Deficient"
        elif value < opt_min:
            status = "Borderline Low"
            sev = 5.0
        elif value <= opt_max:
            status = "Optimal"
            sev = 0.0
        elif value < high:
            status = "Borderline High"
            sev = 4.0
        else:
            status = "Excess"
            sev = 8.0

        results.append({"nutrient": nut, "value": value, "status": status, "severity_score": float(sev)})

    return results


# ============================================================
# SOIL CONDITION RULES
# ============================================================

def assess_soil_condition(row: dict):
    """
    Basic soil condition rules. Returns list of issues dicts with 'deficiency' and 'severity_score'.
    """
    issues = []
    try:
        ph = float(row.get("Soil_Ph", np.nan))
    except Exception:
        ph = np.nan
    try:
        ec = float(row.get("Ec_Dsm", np.nan))
    except Exception:
        ec = np.nan
    try:
        oc = float(row.get("Organic_Carbon_Percent", np.nan))
    except Exception:
        oc = np.nan

    if not np.isnan(ph):
        if ph < 5.5:
            issues.append({"deficiency": "Soil_Acidic", "severity_score": 8})
        elif ph > 8.2:
            issues.append({"deficiency": "Soil_Alkaline", "severity_score": 8})

    if not np.isnan(ec) and ec >= 4.0:
        issues.append({"deficiency": "EC_High", "severity_score": 7})

    if not np.isnan(oc) and oc < 0.3:
        issues.append({"deficiency": "Low_Organic_Carbon", "severity_score": 5})

    return issues


# ============================================================
# BUILD DEFICIENCY LIST
# ============================================================

def build_deficiency_list(row: dict, nutrient_thresholds: pd.DataFrame):
    """
    Combines nutrient and soil issues into a single list of deficiencies.
    """
    result = []

    for n in assess_nutrient_status(row, nutrient_thresholds):
        if n["status"] not in ("Optimal", "Unknown"):
            result.append({
                "type": "Nutrient",
                "deficiency": n["nutrient"],
                "severity": n["severity_score"]
            })

    for s in assess_soil_condition(row):
        result.append({"type": "Soil", "deficiency": s["deficiency"], "severity": s["severity_score"]})

    return result


# ============================================================
# TREATMENT SCORE
# ============================================================

def score_treatment(rec: pd.Series, growth_stage: str, phi_class: str, severity: float) -> float:
    """
    Score how important a treatment is: severity scaled by stage relevance and phi multiplier.
    rec must contain a Stage_Priority field (string).
    """
    if rec is None or rec.empty:
        return 0.0

    # normalize stage & label to lowercase for matching keys
    growth_stage_norm = (str(growth_stage) or "").strip().lower()
    stage_map = STAGE_RELEVANCE.get(growth_stage_norm, {})
    label = str(rec.get("Stage_Priority", "")).strip().lower()

    # default weight if not found
    weight = stage_map.get(label, 0.7)

    phi_multiplier = {
        "very healthy": 0.6, "healthy": 0.8, "at risk but recoverable": 1.0,
        "moderate stress": 1.1, "severe stress": 1.2,
    }.get(str(phi_class).strip().lower(), 1.0)

    try:
        sev = float(severity)
    except Exception:
        sev = 0.0

    return (sev / 10.0) * float(weight) * float(phi_multiplier)


# ============================================================
# MAIN TREATMENT ENGINE
# ============================================================

def generate_treatment_plan(row: dict):
    """
    row: dict-like (can be pandas Series or plain dict) that contains at least:
      - Crop_Name, Growth_Stage, PHI, SQI and nutrient & soil columns used earlier.
    Returns dict with SQI, SQI_Class, PHI, PHI_Class, Treatments (list).
    """
    nutrient_thresholds, crop_req, treatment_recs, pest_actions = load_reference_tables()

    # Ensure row is a dict for easier .get access
    if isinstance(row, pd.Series):
        row = row.to_dict()

    growth_stage = str(row.get("Growth_Stage", "")).strip().lower()
    phi_class = classify_phi(float(row.get("PHI", 0.0)))

    deficiencies = build_deficiency_list(row, nutrient_thresholds)

    actions = []

    if treatment_recs is None or treatment_recs.empty:
        # no treatment table — return empty plan but include classes
        return {
            "SQI": row.get("SQI", np.nan),
            "SQI_Class": classify_sqi(float(row.get("SQI", 0.0))),
            "PHI": row.get("PHI", np.nan),
            "PHI_Class": phi_class,
            "Treatments": []
        }

    # Normalize treatment_recs 'Deficiency' column for matching
    tr = treatment_recs.copy()
    if "Deficiency" in tr.columns:
        tr["_def_norm"] = tr["Deficiency"].astype(str).str.strip().str.lower()
    else:
        tr["_def_norm"] = ""

    for d in deficiencies:
        search_def = str(d["deficiency"]).strip().lower()
        rec_rows = tr[tr["_def_norm"] == search_def]
        if rec_rows.empty:
            # try exact capitalization fallback (maybe soil issues stored differently)
            rec_rows = tr[tr["_def_norm"].str.contains(search_def, na=False)]

        if rec_rows.empty:
            continue

        # pick first matching rec row
        rec = rec_rows.iloc[0]

        # dose: attempt to pick column suited to soil texture if available, else fallback
        dose = None
        # prefer texture-specific dose columns if present
        for col_try in ("Soil_Loam_Dose", "Soil_Clay_Dose", "Soil_Sand_Dose", "Dose", "Recommended_Dose"):
            if col_try in rec.index and not pd.isna(rec.get(col_try)):
                dose = rec.get(col_try)
                break
        if pd.isna(dose):
            dose = None

        score = score_treatment(rec, growth_stage, phi_class, d["severity"])

        
        actions.append({
            "issue": d["deficiency"],
            "fertilizer": rec.get("Fertilizer", "Unknown"),
            "dose": dose if dose else "N/A",
            "priority_score": round(score, 3),
            "notes": rec.get("Notes", "")
        })


    # sort by score and return top 5
    actions = sorted(actions, key=lambda x: x["Score"], reverse=True)[:5]

    

    return {
        "SQI": row.get("SQI", np.nan),
        "SQI_Class": classify_sqi(float(row.get("SQI", 0.0))),
        "PHI": row.get("PHI", np.nan),
        "PHI_Class": phi_class,
        "Treatments": actions
    }


# ============================================================
# PRETTY OUTPUT
# ============================================================

def print_plan(plan: dict):
    print("\n DIAGNOSTIC SUMMARY \n")

    # -----------------------------------------
    # 1. Show only the class labels (no numbers)
    # -----------------------------------------
    sqi_class = plan.get("SQI_Class", "Unknown")
    phi_class = plan.get("PHI_Class", "Unknown")
    print()
    print(f"• Soil Quality Index (SQI): {sqi_class}")
    print(f"• Crop Health Index (CHI): {phi_class}")
    print()
    print()


    # -----------------------------------------
    # 2. If no treatments required
    # -----------------------------------------
    treatments = plan.get("Treatments", [])
    if not treatments:
        print()
        print("🌱 No treatment required.\nThe soil and plant conditions appear stable.")
        print()
        return

    # -----------------------------------------
    # 3. Show treatment recommendations
    # -----------------------------------------
    print("📌 Recommended Actions:\n")

    # Sort by priority score (highest → lowest)
    treatments_sorted = sorted(
        treatments, key=lambda x: x.get("Score", 0), reverse=True
    )

    for i, t in enumerate(treatments_sorted, start=1):
        print()
        print(f"{i}. {t.get('Issue', 'Unknown Issue')}")
        print("\n")
        print(f"   → Fertilizer: {t.get('Fertilizer', 'N/A')}")
        print("\n")
        print(f"   → Dose: {t.get('Dose', 'N/A')} kg/acre")
        print("\n")
        print(f"   → Notes: {t.get('Notes', 'N/A')}")

    print("\n")
    print()

