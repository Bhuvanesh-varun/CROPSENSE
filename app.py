from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from flask_cors import CORS
from datetime import datetime, timezone, timedelta
from markupsafe import escape
from flask import send_file
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from utils.knn_imputer import infer_soil_defaults_knn
from dotenv import load_dotenv
load_dotenv()

# Razorpay
import razorpay
import hmac
import hashlib


# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore, auth as firebase_auth



def generate_simple_user_id(uid):
    return "USR-" + uid[:5].upper()

def safe_json(obj):
    import json
    return json.loads(json.dumps(obj, default=str))




# helper function for time
def ts_to_dt_and_epoch(ts):
    if not ts:
        return None, 0

    try:
        if hasattr(ts, "timestamp"):
            dt_utc = datetime.fromtimestamp(ts.timestamp(), tz=timezone.utc)
        else:
            return None, 0

        dt_ist = dt_utc.astimezone(IST)
        return dt_ist, dt_ist.timestamp()

    except:
        return None, 0



# ---------------------------------------------------
# FLASK APP SETUP
# ---------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
CORS(app, supports_credentials=True)



# ---------------------------------------------------
# FIREBASE SETUP
# ---------------------------------------------------
SERVICE_ACCOUNT_PATH = os.path.join(os.getcwd(), "cropsense-firebase-adminsdk.json")

if not os.path.exists(SERVICE_ACCOUNT_PATH):
    raise FileNotFoundError("❌ Missing cropsense-firebase-adminsdk.json in project root!")

cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)



# def init_firebase():
#     firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

#     if not firebase_json:
#         raise RuntimeError("❌ FIREBASE_SERVICE_ACCOUNT_JSON env variable not set")

#     try:
#         service_account_info = json.loads(firebase_json)
#     except json.JSONDecodeError as e:
#         raise RuntimeError("❌ Invalid Firebase service account JSON") from e

#     # Fix private key newlines (important for env vars)
#     if "private_key" in service_account_info:
#         service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

#     if not firebase_admin._apps:
#         cred = credentials.Certificate(service_account_info)
#         firebase_admin.initialize_app(cred)


# # 🔥 CALL THIS ON APP STARTUP
# init_firebase()

db = firestore.client()

@app.context_processor
def inject_user():
    user = None
    if session.get("user_id"):
        doc = db.collection("users").document(session["user_id"]).get()
        if doc.exists:
            user = doc.to_dict()

            # 🔒 HARD TYPE SAFETY FOR QUOTA
            quota = user.get("quota")
            if quota:
                try:
                    quota["total"] = int(quota.get("total", 0))
                    quota["used"] = int(quota.get("used", 0))
                except Exception:
                    quota["total"] = 0
                    quota["used"] = 0

    return dict(current_user=user)




# ---------------------------------------------------
# RAZORPAY SETUP
# ---------------------------------------------------


RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

PLANS = {
    "single":  {"price": 900,  "label": "Single Run"},   # ₹9 = 900 paise
    "starter": {"price": 4900, "label": "Starter"},      # ₹50
    "pro":     {"price": 9900,"label": "Pro"}           # ₹100
}



# ---------------------------------------------------
# ADMIN LOGIN CREDENTIALS
# ---------------------------------------------------
ADMIN_EMAIL = "admin@cropsense.com"
ADMIN_PASSWORD = "admin@cropsense@123"


def admin_required():
    if session.get("role") != "admin":
        return redirect(url_for("index"))


# ---------------------------------------------------
# LOGOUT
# ---------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ---------------------------------------------------
# HELPER: FETCH USER HISTORY (Aligned with Core)
# ---------------------------------------------------
def get_user_history(uid):

    user_doc = db.collection("users").document(uid).get().to_dict()
    plan = user_doc.get("plan", "free")

    if plan in ["free", "single"]:
        return []   # no history access

    history = []

    try:
        docs = (
            db.collection("users")
            .document(uid)
            .collection("predictions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )

        for d in docs:
            rec = d.to_dict() or {}

            # Convert Firestore timestamp → IST datetime + epoch
            ts_raw = rec.get("timestamp")
            ts_dt, ts_epoch = ts_to_dt_and_epoch(ts_raw)

            history.append({
                "id": d.id,
                "timestamp": ts_dt.strftime("%d %b %Y, %I:%M %p") if ts_dt else None,
                "timestamp_epoch": ts_epoch,

                "sqi": rec.get("sqi"),
                "phi": rec.get("phi"),
                "sqi_text": rec.get("sqi_text"),
                "phi_text": rec.get("phi_text"),

                # Treatment outputs
                "plan_text": rec.get("plan_text"),
                "plan_html": rec.get("plan_html"),

                # Raw inputs (useful for re-run / audit)
                "input_json": rec.get("input_json", {}),
                "crop": (rec.get("inputs_json", {}) or {}).get("Crop_Name", "Unknown Crop")
            })

    except Exception as e:
        print("🔥 Error reading user history:", e)

    return history


# ---------------------------------------------------------
# IMPORT YOUR TREATMENT ENGINE
# ---------------------------------------------------------
from utils.treatment_engine import generate_treatment_plan, print_plan

# ---------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------
SQI_MODEL_PATH = "models/SQI_full_pipeline.pkl"
PHI_MODEL_PATH = "models/PHI_full_pipeline.pkl"
TRAIN_DATA_PATH = "data/data_validated.csv"

sqi_model = joblib.load(SQI_MODEL_PATH)
phi_model = joblib.load(PHI_MODEL_PATH)
df_train = pd.read_csv(TRAIN_DATA_PATH)

# ---------------------------------------------------------
# CLASSIFIERS FOR SQI + PHI
# ---------------------------------------------------------
def classify_sqi(sqi_value: float) -> str:
    if sqi_value <= 1.4:
        return "Very Poor"
    elif sqi_value <= 2.4:
        return "Poor"
    elif sqi_value <= 3.4:
        return "Moderate"
    elif sqi_value <= 4.2:
        return "Good"
    return "Excellent"


def classify_phi(phi_value: float) -> str:
    if phi_value <= 3.0:
        return "Severe Stress"
    elif phi_value <= 5.0:
        return "Moderate Stress"
    elif phi_value <= 7.5:
        return "At Risk but Recoverable"
    elif phi_value <= 9.0:
        return "Healthy"
    return "Very Healthy"


def format_plan_html(plan: dict) -> str:
    """
    Convert the treatment plan dict into a readable HTML fragment.
    Uses simple escaping to avoid injection.
    """
    parts = []
    # Header / final message
    final_message = plan.get("final_message") or ""
    if final_message:
        parts.append(f"<h4 class='fw-bold'>{escape(final_message)}</h4>")

    # NPK
    N = plan.get("N")
    P = plan.get("P")
    K = plan.get("K")
    if any(v is not None for v in (N, P, K)):
        parts.append(f"<p><strong>Estimated N / P / K (kg/ha):</strong> {escape(str(N))} / {escape(str(P))} / {escape(str(K))}</p>")

    # deficiencies
    deficiencies = plan.get("deficiencies") or []
    if deficiencies:
        parts.append("<h5 class='mt-3'>Detected Deficiencies</h5><ul>")
        for d in deficiencies:
            # expecting d like {'deficiency':'Nitrogen','severity':'High','notes':'...'}
            name = escape(str(d.get("deficiency") or d.get("name") or "Unknown"))
            sev = escape(str(d.get("severity") or ""))
            notes = escape(str(d.get("notes") or ""))
            parts.append(f"<li><b>{name}</b> — {sev} <div class='text-muted small'>{notes}</div></li>")
        parts.append("</ul>")

    # treatments
    treatments = safe_json(plan.get("treatments", [])) or []
    if treatments:
        parts.append("<h5 class='mt-3'>Treatment Recommendations</h5><ol>")
        for t in treatments:
            # expecting t like {'Fertilizer':'Urea', 'Dose':'50 kg/ha','Notes':'apply...'}
            fert = escape(str(t.get("Fertilizer") or t.get("treatment") or "Treatment"))
            dose = escape(str(t.get("Dose") or t.get("dose") or ""))
            notes = escape(str(t.get("Notes") or t.get("notes") or ""))
            parts.append(f"<li><b>{fert}</b> — {dose}<div class='text-muted small'>{notes}</div></li>")
        parts.append("</ol>")

    # fallback if nothing
    if not parts:
        return "<p>No plan items generated.</p>"

    return "\n".join(parts)




# ---------------------------------------------------------
# OUTLIER COLUMNS (set to 0)
# ---------------------------------------------------------
outlier_cols = [
    "Ec_Dsm_low_outlier", "Ec_Dsm_high_outlier",
    "Available_N_Kg_Ha_low_outlier", "Available_N_Kg_Ha_high_outlier",
    "Available_S_Kg_Ha_low_outlier", "Available_S_Kg_Ha_high_outlier",
    "Sunlight_Hours_Per_Day_low_outlier", "Sunlight_Hours_Per_Day_high_outlier",
    "No_Of_Irrigations_Since_Sowing_low_outlier",
    "No_Of_Irrigations_Since_Sowing_high_outlier",
    "SQI_low_outlier", "SQI_high_outlier",
    "Plant_Height_Cm_low_outlier", "Plant_Height_Cm_high_outlier",
    "Pesticide_Dosage_Ml_Per_Acre_low_outlier",
    "Pesticide_Dosage_Ml_Per_Acre_high_outlier",
    "Fungicide_Sprays_Last_30_Days_low_outlier",
    "Fungicide_Sprays_Last_30_Days_high_outlier",
    "height_mean_stage_low_outlier", "height_mean_stage_high_outlier",
    "height_std_crop_low_outlier", "height_std_crop_high_outlier",
]


# ---------------------------------------------------------
# MAIN ROUTE FOR FRONTEND
# ---------------------------------------------------------
@app.post("/get_recommendation")
def get_recommendation():
    
    if "user_id" not in session:
        return jsonify({
            "status": "unauthorized",
            "message": "Login required"
        }), 401
    
    import time
    start = time.time()

    # --------------------------------------------------
    # STEP 2.1 — CHECK USER QUOTA BEFORE AI RUN
    # --------------------------------------------------
    user_id = session.get("user_id")
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()

    if not user_doc.exists:
        return jsonify({
            "status": "error",
            "message": "User record not found"
        }), 400

    user_data = user_doc.to_dict()
    quota = user_data.get("quota", {})
    used = quota.get("used", 0)
    total = quota.get("total", 0)

    if used >= total:
        return jsonify({
            "status": "limit_reached",
            "message": "Trial limit reached. Please upgrade your plan."
        }), 403

    
    try:
        data = request.get_json(force=True)

        # -----------------------------
        # Normalize frontend inputs
        # -----------------------------
        base_row = {
            "Crop_Name": data.get("crop"),
            "Previous_Crop": data.get("previousCrop"),
            "Soil_Type": data.get("soilType"),
            "Soil_Texture_Class": data.get("soilTexture"),
            "Growth_Stage": data.get("growthStage"),

            "Irrigation_Type": data.get("irrigationType"),
            "Current_Soil_State": data.get("irrigationStatus"),
            "No_Of_Irrigations_Since_Sowing": int(data.get("irrigationCount", 0)),

            "Leaf_Colour": data.get("leafColor"),
            "Leaf_Yellowing_Percent": float(data.get("leafYellowPercent", 0)),
            "Leaf_Spot_Severity": float(data.get("spots", 0)),
            "Pest_Incidence": data.get("pests"),

            # Fertilizer
            "Last_Fertilized_15Days": "yes" if data.get("usedFertilizer") == "Yes" else "no",
            "Fertilizer_Type_Last_Used": (data.get("fertilizerType") or "none"),
            "Last_Fertilizer_Dosage": float(data.get("fertilizerQty", 0)),

            # Pesticide
            "Pesticide_Type_Last_Used": (data.get("pesticideType") or "none"),
            "Pesticide_Dosage_Ml_Per_Acre": float(data.get("pesticideQty", 0)),

            # Fungicide
            "Fungicide_Sprays_Last_30_Days": int(data.get("fungSprays", 0)),
        }


        # -----------------------------
        # Infer missing numeric fields
        # -----------------------------
        inferred = infer_soil_defaults_knn(df_train, base_row, k=20)
        full_row = {**base_row, **inferred}

        # Add static stats expected by models
        full_row.update({
            "height_mean_stage": 40.0,
            "height_std_stage": 8.0,
            "height_mean_crop": 42.0,
            "height_std_crop": 9.0,
        })

        # Outlier flags
        for col in outlier_cols:
            full_row[col] = 0

        sample_df = pd.DataFrame([full_row])

        # -----------------------------
        # Predict SQI & PHI
        # -----------------------------
        sqi_value = float(sqi_model.predict(sample_df)[0])
        sample_df["SQI"] = sqi_value

        phi_value = float(phi_model.predict(sample_df)[0])
        sample_df["PHI"] = phi_value

        sample_df.to_csv("input_full.csv", index=False)  # for debugging

        # -----------------------------
        # Generate treatment plan
        # -----------------------------
        plan = generate_treatment_plan(sample_df.iloc[0])
        print_plan(plan)

        # Capture human-readable plan text
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_plan(plan)
        plan_text = buf.getvalue().strip()

        # -----------------------------
        # Persist to session (CORE)
        # -----------------------------
        session["ai_last_run"] = {
            "inputs": base_row,
            "sqi": round(sqi_value, 2),
            "phi": round(phi_value, 2),
            "sqi_class": plan.get("SQI_Class"),
            "phi_class": plan.get("PHI_Class"),
            "treatments": safe_json(plan.get("treatments", [])),
            "plan_text": plan_text,
        }

        # -----------------------------
        # Persist prediction to Firestore (USER DASHBOARD CORE)
        # -----------------------------
        user_id = session.get("user_id")

        if user_id:
            db.collection("users") \
            .document(user_id) \
            .collection("predictions") \
            .add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "sqi": round(sqi_value, 2),
                "phi": round(phi_value, 2),
                "sqi_class": plan.get("SQI_Class"),
                "phi_class": plan.get("PHI_Class"),
                "inputs": base_row,
                "crop": base_row.get("Crop_Name"),
                "treatments": safe_json(plan.get("treatments", [])),
                "plan_text": plan_text,
                "plan_html": format_plan_html(plan)
            })

        # -----------------------------
        # Return clean API response
        # -----------------------------
        user_ref.update({
            "quota.used": firestore.Increment(1)
        })

        print("⏱️ get_recommendation took", round(time.time() - start, 2), "seconds")

        return jsonify({
            "status": "success",
            "data": session["ai_last_run"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    



# ---------------------------------------------------------
# RENDER YOUR HTML PAGES
# ---------------------------------------------------------
@app.route("/", endpoint="index")
def index():
    return render_template("index.html")


@app.route("/about", endpoint="about_us")
def about_page():
    return render_template("aboutus.html")

@app.route("/contact", endpoint="contact_us")
def contact_page():
    return render_template("contactus.html")

@app.route("/recommendations", endpoint="recommendations")
def recommendations_page():
    return render_template("recommendations.html")

@app.route("/plans")
def plans():
    user = None
    if session.get("user_id"):
        user = db.collection("users").document(session["user_id"]).get().to_dict()
    return render_template("plans.html", user=user)



# ---------------------------------------------------------
# Persist recommendation when user navigates away & comes back
# ---------------------------------------------------------

@app.get("/last_recommendation")
def last_recommendation():
    if "ai_last_run" not in session:
        return jsonify({"status": "empty"})
    return jsonify({
        "status": "success",
        "data": session["ai_last_run"]
    })


# ---------------------------------------------------
# TIMEZONE (DEFINE ONCE – CORE ALIGNED)
# ---------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))



# ---------------------------------------------------
# ADMIN DASHBOARD
# ---------------------------------------------------
@app.route("/admin/dashboard")
def admin_dashboard():
    guard = admin_required()
    if guard:
        return guard

    users = []
    free_users, paid_users = [], []
    total_predictions = 0

    for doc in db.collection("users").stream():
        d = doc.to_dict()
        user = {
            "id": doc.id,
            "name": d.get("name", "—"),
            "email": d.get("email", "—"),
            "role": d.get("role", "user"),
            "plan": d.get("plan", "free")
        }
        users.append(user)

        if user["plan"] == "free":
            free_users.append(user)
        else:
            paid_users.append(user)

        # count predictions
        try:
            preds = db.collection("users").document(doc.id)\
                     .collection("predictions").stream()
            total_predictions += len(list(preds))
        except:
            pass

    return render_template(
        "admin_dashboard.html",
        users=users,
        free_users=free_users,
        paid_users=paid_users,
        total_predictions=total_predictions
    )


# ---------------------------------------------------
# ADMIN – USER HISTORY
# ---------------------------------------------------
@app.route("/admin/user_history/<user_id>")
def admin_user_history(user_id):
    guard = admin_required()
    if guard:
        return guard

    user_doc = db.collection("users").document(user_id).get()
    if not user_doc.exists:
        flash("User not found", "danger")
        return redirect(url_for("admin_dashboard"))

    user = user_doc.to_dict()
    history = get_user_history(user_id)

    return render_template(
        "user_history.html",
        history=history,
        is_admin=True,
        user=user
    )



# ---------------------------------------------------
# ADMIN – TOGGLE ADMIN ROLE
# ---------------------------------------------------
@app.route("/admin/toggle_admin/<user_id>", methods=["POST"])
def admin_toggle_admin(user_id):
    guard = admin_required()
    if guard:
        return guard

    try:
        ref = db.collection("users").document(user_id)
        data = ref.get().to_dict()

        if not data:
            flash("User not found.", "danger")
            return redirect(url_for("admin_dashboard"))

        role = data.get("role", "user")

        if role != "admin":
            firebase_auth.set_custom_user_claims(user_id, {"role": "admin"})
            ref.update({"role": "admin"})
            flash("User promoted to admin!", "success")
        else:
            firebase_auth.set_custom_user_claims(user_id, {"role": "user"})
            ref.update({"role": "user"})
            flash("User demoted to user.", "warning")

    except Exception as e:
        print("🔥 Admin toggle error:", e)
        flash("Operation failed.", "danger")

    return redirect(url_for("admin_dashboard"))


# ---------------------------------------------------
# ADMIN – DELETE USER
# ---------------------------------------------------
@app.route("/admin/delete/<user_id>", methods=["POST"])
def admin_delete_user(user_id):
    guard = admin_required()
    if guard:
        return guard

    if user_id == session.get("user_id"):
        flash("You cannot delete yourself.", "danger")
        return redirect(url_for("admin_dashboard"))

    try:
        firebase_auth.delete_user(user_id)

        for log in db.collection("users").document(user_id) \
            .collection("login_logs").stream():
            log.reference.delete()

        for pred in db.collection("users").document(user_id) \
            .collection("predictions").stream():
            pred.reference.delete()

        db.collection("users").document(user_id).delete()
        flash("User deleted successfully.", "success")

    except Exception as e:
        print("🔥 Delete user error:", e)
        flash("Deletion failed.", "danger")

    return redirect(url_for("admin_dashboard"))



@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/signup")
def signup():
    return render_template("signup.html")



# ---------------------------------------------------
# SESSION LOGIN (CORE-ALIGNED)
# ---------------------------------------------------
@app.route("/session_login", methods=["POST"])
def session_login():
    try:
        data = request.get_json(silent=True)
        if not data or "idToken" not in data:
            return jsonify({"ok": False, "error": "Missing idToken"}), 400

        id_token = data["idToken"]

        decoded = firebase_auth.verify_id_token(id_token)

        firebase_uid = decoded.get("uid")
        email = decoded.get("email", "")
        name = decoded.get("name") or email.split("@")[0]

        if not firebase_uid or not email:
            return jsonify({"ok": False, "error": "Invalid Firebase token"}), 400

        # 🔍 Look for user in Firestore
        user_query = db.collection("users") \
            .where("uid", "==", firebase_uid) \
            .limit(1) \
            .stream()

        user_doc = next(user_query, None)

        # 🆕 AUTO-CREATE USER IF NOT EXISTS (IMPORTANT)
        if not user_doc:
            user_ref = db.collection("users").document()
            user_ref.set({
                "uid": firebase_uid,
                "email": email,
                "name": name,
                "role": "user",
                "created_at": firestore.SERVER_TIMESTAMP
            })
            user_id = user_ref.id
        else:
            user_id = user_doc.id
        
        # --------------------------------------------------
        # STEP 1.1 — ENSURE QUOTA & PLAN EXIST
        # --------------------------------------------------
        user_ref = db.collection("users").document(user_id)
        user_data = user_ref.get().to_dict()

        if "quota" not in user_data:
            user_ref.update({
                "quota": {
                    "total": 3,
                    "used": 0
                },
                "plan": "free"
            })


        # 🧠 Session
        session["user"] = name
        session["email"] = email
        session["user_id"] = user_id
        session["role"] = (
            user_doc.to_dict().get("role", "user")
            if user_doc else "user"
        )
        print("✅ Session set:", dict(session))

        # 📝 Login log
        db.collection("users").document(user_id) \
            .collection("login_logs").add({
                "timestamp": firestore.SERVER_TIMESTAMP,
                "event": "login",
                "email": email
            })

        redirect_url = (
            url_for("admin_dashboard")
            if session.get("role") == "admin"
            else url_for("recommendations", login="success")
        )

        return jsonify({
            "ok": True,
            "redirect": redirect_url
        })


    except Exception as e:
        print("🔥 Session login error:", e)
        return jsonify({"ok": False, "error": "Session login failed"}), 400


# ---------------------------------------------------
# USER HISTORY (CORE-ALIGNED)
# ---------------------------------------------------

@app.route("/dashboard")
def user_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    history = get_user_history(session["user_id"])
    return render_template("dashboard.html", history=history)

@app.route("/my_history")
def my_history():
    if "user_id" not in session:
        return redirect(url_for("index"))

    user_doc = db.collection("users").document(session["user_id"]).get()
    if not user_doc.exists:
        return redirect(url_for("index"))

    user = user_doc.to_dict()
    plan = user.get("plan", "free")

    # 🚫 Block free & single users completely
    if plan in ["free", "single"]:
        return render_template(
            "upgrade_required.html",
            feature="History Access",
            plan=plan
        )

    history = get_user_history(session["user_id"])
    return render_template(
        "user_history.html",
        history=history,
        is_admin=False
    )



@app.route("/prediction/<prediction_id>")
def view_prediction(prediction_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    doc = db.collection("users") \
        .document(session["user_id"]) \
        .collection("predictions") \
        .document(prediction_id) \
        .get()

    if not doc.exists:
        flash("Prediction not found", "danger")
        return redirect(url_for("my_history"))

    return render_template(
        "prediction_view.html",
        prediction=doc.to_dict()
    )



@app.route("/prediction/download/<prediction_id>")
def download_prediction(prediction_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    doc = (
        db.collection("users")
        .document(session["user_id"])
        .collection("predictions")
        .document(prediction_id)
        .get()
    )

    if not doc.exists:
        flash("Prediction not found", "danger")
        return redirect(url_for("my_history"))

    data = doc.to_dict()

    # ---------------- PDF GENERATION ----------------
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>CropSense AI Prediction Report</b>", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"<b>SQI:</b> {data.get('sqi')}", styles["Normal"]))
    content.append(Paragraph(f"<b>PHI:</b> {data.get('phi')}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("<b>Treatment Plan:</b>", styles["Heading2"]))
    content.append(Spacer(1, 8))

    plan_text = data.get("plan_text", "No recommendations available.")
    for line in plan_text.split("\n"):
        content.append(Paragraph(line, styles["Normal"]))

    pdf.build(content)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="CropSense_Prediction_Report.pdf",
        mimetype="application/pdf"
    )



@app.post("/prediction/delete/<prediction_id>")
def delete_prediction(prediction_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    db.collection("users") \
      .document(session["user_id"]) \
      .collection("predictions") \
      .document(prediction_id) \
      .delete()

    flash("Prediction deleted", "success")
    return redirect(url_for("my_history"))





# ---------------------------------------------------
# CREATE ORDER (UNCHANGED – VERIFIED OK)
# ---------------------------------------------------
@app.route("/create_order", methods=["POST"])
def create_order():
    if "user_id" not in session:
        return jsonify({"ok": False, "error": "Not logged in"}), 401

    data = request.get_json()
    plan = data.get("plan")

    if plan not in PLANS:
        return jsonify({"ok": False, "error": "Invalid plan"}), 400

    try:
        order = razorpay_client.order.create({
            "amount": PLANS[plan]["price"],
            "currency": "INR",
            "receipt": f"rcpt_{session['user_id']}",
            "payment_capture": 1,
            "notes": {"plan": plan, "user_id": session["user_id"]}
        })
        return jsonify({"ok": True, "order": order, "key_id": RAZORPAY_KEY_ID})

    except Exception as e:
        print("🔥 Razorpay error:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------
# VERIFY PAYMENT (CORE-ALIGNED)
# ---------------------------------------------------
@app.route("/verify_payment", methods=["POST"])
def verify_payment():
    data = request.get_json()

    # 🔐 Verify Razorpay signature
    msg = f"{data['razorpay_order_id']}|{data['razorpay_payment_id']}".encode()
    expected = hmac.new(
        RAZORPAY_KEY_SECRET.encode(),
        msg,
        hashlib.sha256
    ).hexdigest()

    if expected != data.get("razorpay_signature"):
        return jsonify({"ok": False, "error": "Signature mismatch"}), 400

    uid = session.get("user_id")
    plan = data.get("plan")

    if not uid or not plan:
        return jsonify({"ok": False, "error": "Invalid user or plan"}), 400

    # 🎯 PLAN → QUOTA MAPPING
    PLAN_CONFIG = {
        "single":  {"plan": "single",  "quota": 1},
        "starter": {"plan": "starter", "quota": 10},
        "pro":     {"plan": "pro",     "quota": 20},
    }

    if plan not in PLAN_CONFIG:
        return jsonify({"ok": False, "error": "Unknown plan"}), 400

    config = PLAN_CONFIG[plan]

    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get().to_dict()

    current_used = user_doc.get("quota", {}).get("used", 0)

    # ✅ APPLY PLAN + RESET QUOTA TOTAL (KEEP USED)
    user_ref.update({
        "plan": config["plan"],
        "quota": {
            "total": config["quota"],
            "used": current_used
        },
        "subscription": {
            "plan": config["plan"],
            "date": firestore.SERVER_TIMESTAMP
        }
    })

    # 🧾 STORE PAYMENT RECORD
    user_ref.collection("payments").add({
        "plan": config["plan"],
        "order_id": data["razorpay_order_id"],
        "payment_id": data["razorpay_payment_id"],
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    # 🧠 Update session (IMPORTANT)
    session["plan"] = config["plan"]

    return jsonify({"ok": True})


@app.route("/check_subscription")
def check_subscription():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False, "authenticated": False})

    doc = db.collection("users").document(uid).get()
    if not doc.exists:
        return jsonify({"ok": True, "authenticated": True, "subscription": None})

    return jsonify({
        "ok": True,
        "authenticated": True,
        "subscription": doc.to_dict().get("subscription")
    })


# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

