import os
import joblib
import numpy as np
import pandas as pd
import logging

#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder, StandardScaler
#from sklearn.multioutput import MultiOutputRegressor


# ---------------------------------------------------------
# AUTO-FILL NUTRIENTS BASED ON TRAIN DATA
# ---------------------------------------------------------
NUTRIENT_COLS = [
    "Soil_Ph",
    "Ec_Dsm",
    "Organic_Carbon_Percent",
    "Available_N_Kg_Ha",
    "Available_P_Kg_Ha",
    "Available_K_Kg_Ha",
    "Available_S_Kg_Ha",
    "Available_Zn_Ppm",
    "Available_B_Ppm",
    "Available_Fe_Ppm",
    "Available_Mn_Ppm",
    "Available_Cu_Ppm",
]



# features to use for similarity
_KNN_FEATURES = [
    "Crop_Name", "Soil_Type", "Soil_Texture_Class", "Irrigation_Type", "Previous_Crop",
    "Growth_Stage", "Leaf_Colour", "Pest_Incidence", "Last_Fertilized_15Days", "Fertilizer_Type_Last_Used",
    "Pesticide_Type_Last_Used", "Fungicide_Sprays_Last_30_Days"
]

# nutrients we want to predict
_KNN_TARGETS = [
    "Soil_Ph",
    "Ec_Dsm",
    "Organic_Carbon_Percent",
    "Plant_Height_Cm",
    "Temperature_Avg",
    "Rainfall_Last_7Days",
    "Humidity_Percent",
    "Sunlight_Hours_Per_Day",
    "Available_N_Kg_Ha",
    "Available_P_Kg_Ha",
    "Available_K_Kg_Ha",
    "Available_S_Kg_Ha",
    "Available_Zn_Ppm",
    "Available_B_Ppm",
    "Available_Fe_Ppm",
    "Available_Mn_Ppm",
    "Available_Cu_Ppm",

]

import logging
logger = logging.getLogger(__name__)


# cache filenames (optional)
_KNN_PIPE_CACHE = "models/knn_imputer_pipeline.joblib"
_KNN_MODEL_CACHE = "models/knn_imputer_model.joblib"


# def fit_knn_imputer(df_train,
#                     k=20,
#                     min_rows_for_fit=1,
#                     force_refit=False):
#     """
#     Fit + cache a preprocessing pipeline and KNN multioutput regressor.
#     Returns (preprocessor, knn_model).

#     Fixes applied:
#     - Ensure columns exist before accessing dtype.
#     - Construct categorical/numeric lists robustly.
#     - Build ColumnTransformer only with present transformer lists.
#     - Use sparse_output=False for OneHotEncoder (compat with sklearn >=1.0).
#     - Cache load + fallback handling kept.
#     """
    
#     if (not force_refit) and os.path.exists(_KNN_PIPE_CACHE) and os.path.exists(_KNN_MODEL_CACHE):
#         try:
#             preproc = joblib.load(_KNN_PIPE_CACHE)
#             knn = joblib.load(_KNN_MODEL_CACHE)
#             return preproc, knn
#         except Exception:
#             # if cache load fails, continue to refit
#             pass

#     # Keep only rows with non-null targets
#     df_fit = df_train.dropna(subset=_KNN_TARGETS, how='any').copy()
#     if len(df_fit) < min_rows_for_fit:
#         # not enough rows to fit; raise or fallback to median approach
#         raise ValueError(f"Not enough training rows to fit KNN imputer (have {len(df_fit)})")

#     # Ensure all features exist in df_fit (create with NaN if missing) BEFORE dtype checks
#     for c in _KNN_FEATURES:
#         if c not in df_fit.columns:
#             df_fit[c] = np.nan

#     # Determine categorical vs numeric robustly
#     # Treat explicitly known categorical fields as categorical
#     explicit_cats = {
#         "Crop_Name", "Soil_Type", "Soil_Texture_Class", "Irrigation_Type",
#         "Growth_Stage", "Previous_Crop", "Leaf_Colour", "Pest_Incidence",
#         "Last_Fertilized_15Days", "Fertilizer_Type_Last_Used", "Pesticide_Type_Last_Used"
#     }
#     categorical = [c for c in _KNN_FEATURES if (c in df_fit.columns) and (
#         (df_fit[c].dtype == object) or (c in explicit_cats)
#     )]
#     numeric = [c for c in _KNN_FEATURES if c not in categorical]

#     # Build column transformer safely: only add transformers that have at least one column
#     transformers = []
#     if len(categorical) > 0:
#         # use sparse_output where available
#         try:
#             ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#         except TypeError:
#             # fallback for older sklearn versions
#             ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
#         transformers.append(("cat", ohe, categorical))
#     if len(numeric) > 0:
#         transformers.append(("num", StandardScaler(), numeric))

#     # If nothing to transform, make a passthrough transformer to avoid errors
#     if len(transformers) == 0:
#         # Create a trivial passthrough ColumnTransformer
#         preprocessor = ColumnTransformer(transformers=[], remainder="drop")
#         X = np.zeros((len(df_fit), 0))
#     else:
#         preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

#         # Build X using feature columns; coerce numeric columns to numeric
#         X = df_fit[_KNN_FEATURES].copy()
#         for n in numeric:
#             X[n] = pd.to_numeric(X[n], errors="coerce").fillna(0.0)

#         # fit/transform preprocessor below
#         X = preprocessor.fit_transform(X)

#     # Prepare target Y (multioutput) - ensure column order matches _KNN_TARGETS
#     Y = df_fit[_KNN_TARGETS].astype(float).fillna(0.0).values

#     # KNN multioutput regressor
#     # choose a reasonable n_neighbors but bounded
#     n_neighbors = min(k, max(5, int(len(df_fit) ** 0.5)))
#     base_knn = KNeighborsRegressor(n_neighbors=n_neighbors,
#                                    weights="distance", metric="minkowski", p=2, n_jobs=-1)
#     knn = MultiOutputRegressor(base_knn, n_jobs=-1)
#     # fit on transformed X if we have transformer, else fit on zero-dim array (shouldn't happen)
#     knn.fit(X, Y)

#     # cache for speed
#     try:
#         joblib.dump(preprocessor, _KNN_PIPE_CACHE, compress=3)
#         joblib.dump(knn, _KNN_MODEL_CACHE, compress=3)
#     except Exception:
#         pass

#     return preprocessor, knn


from markupsafe import escape

def infer_soil_defaults_knn(df_train: pd.DataFrame, row: dict,
                            k=20, symptom_penalty=True, force_refit=False):
    """
    Using a KNN regressor (fitted on df_train) predict nutrient values for `row`.
    Returns dict including ALL target entries in _KNN_TARGETS.
    This ensures Plant_Height_Cm, Temperature_Avg, Sunlight_Hours_Per_Day are present.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info("infer_soil_defaults_knn: attempting KNN imputer (k=%s)", k)

    # try to fit or load cached model
    try:
        #preproc, knn = fit_knn_imputer(df_train, k=k, force_refit=force_refit)
        preproc = joblib.load(_KNN_PIPE_CACHE)
        knn = joblib.load(_KNN_MODEL_CACHE)

    except Exception as e:
        logger.warning("infer_soil_defaults_knn: fit_knn_imputer failed — returning medians fallback (%s)", repr(e))
        # fallback: return medians for all _KNN_TARGETS
        med_all = df_train[_KNN_TARGETS].median(numeric_only=True).to_dict()
        out = {t: float(med_all.get(t, np.nan)) for t in _KNN_TARGETS}
        # provide safe defaults if medians are NaN
        if np.isnan(out.get("Plant_Height_Cm", np.nan)):
            out["Plant_Height_Cm"] = 50.0
        if np.isnan(out.get("Temperature_Avg", np.nan)):
            out["Temperature_Avg"] = 28.0
        if np.isnan(out.get("Sunlight_Hours_Per_Day", np.nan)):
            out["Sunlight_Hours_Per_Day"] = 8.5
        return out

    # --- build input DataFrame (single row) with same feature set ---
    input_df = pd.DataFrame([{f: row.get(f, np.nan) for f in _KNN_FEATURES}])

    # coerce expected numeric columns to numeric (if present)
    numeric_candidates = ["Fungicide_Sprays_Last_30_Days",]
    for col in numeric_candidates:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0)

    # Transform using preprocessor; if preprocessor produces error, fallback to medians
    try:
        X_in = preproc.transform(input_df)
    except Exception as e:
        logger.warning("infer_soil_defaults_knn: preproc.transform failed — returning medians fallback (%s)", repr(e))
        med_all = df_train[_KNN_TARGETS].median(numeric_only=True).to_dict()
        out = {t: float(med_all.get(t, np.nan)) for t in _KNN_TARGETS}
        if np.isnan(out.get("Plant_Height_Cm", np.nan)):
            out["Plant_Height_Cm"] = 50.0
        if np.isnan(out.get("Temperature_Avg", np.nan)):
            out["Temperature_Avg"] = 28.0
        if np.isnan(out.get("Sunlight_Hours_Per_Day", np.nan)):
            out["Sunlight_Hours_Per_Day"] = 8.5
        return out

    # get predictions (shape 1 x n_targets)
    try:
        preds = knn.predict(X_in)  # returns np.array
        preds = np.maximum(preds, 0.0)  # prevent negative values
        preds = preds.flatten()
    except Exception as e:
        logger.warning("infer_soil_defaults_knn: knn.predict failed — returning medians fallback (%s)", repr(e))
        med_all = df_train[_KNN_TARGETS].median(numeric_only=True).to_dict()
        out = {t: float(med_all.get(t, np.nan)) for t in _KNN_TARGETS}
        if np.isnan(out.get("Plant_Height_Cm", np.nan)):
            out["Plant_Height_Cm"] = 50.0
        if np.isnan(out.get("Temperature_Avg", np.nan)):
            out["Temperature_Avg"] = 28.0
        if np.isnan(out.get("Sunlight_Hours_Per_Day", np.nan)):
            out["Sunlight_Hours_Per_Day"] = 8.5
        return out

    # preds order matches _KNN_TARGETS
    pred_dict = {t: float(np.round(v, 2)) for t, v in zip(_KNN_TARGETS, preds)}

    # symptom-based adjustments (apply only to nutrient columns)
    if symptom_penalty:
        leaf_colour = (row.get("Leaf_Colour") or "").lower()
        pest = (row.get("Pest_Incidence") or "").lower()

        if "yellow" in leaf_colour or leaf_colour == "yellowish":
            for col in ["Available_N_Kg_Ha", "Available_S_Kg_Ha", "Available_Zn_Ppm"]:
                if col in pred_dict:
                    pred_dict[col] = round(pred_dict[col] * 0.8, 2)

        if pest in ["moderate", "high", "severe"]:
            if "Available_K_Kg_Ha" in pred_dict:
                pred_dict["Available_K_Kg_Ha"] = round(pred_dict["Available_K_Kg_Ha"] * 0.9, 2)

    # ensure all _KNN_TARGETS present, fill with median fallback if missing
    med_all = df_train[_KNN_TARGETS].median(numeric_only=True).to_dict()
    final = {}
    for t in _KNN_TARGETS:
        if t in pred_dict and not (pred_dict[t] is None):
            final[t] = float(pred_dict[t])
        else:
            fallback = med_all.get(t, np.nan)
            if np.isnan(fallback):
                # sensible hard defaults for the three extras
                if t == "Plant_Height_Cm":
                    fallback = 50.0
                elif t == "Temperature_Avg":
                    fallback = 28.0
                elif t == "Sunlight_Hours_Per_Day":
                    fallback = 8.5
                else:
                    fallback = 0.0
            final[t] = float(fallback)

    logger.info("infer_soil_defaults_knn: KNN predict succeeded, returning predicted targets")
    return final
