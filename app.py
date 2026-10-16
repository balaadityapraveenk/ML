import os
import json
import time
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify

from load_data import get_data_summary, get_duplicate_count
from preprocessing import run_preprocessing
from eda import get_eda_summary_data
from LinearRegression import get_linear_regression_summary, predict_sample_salary

from linear_models import run_linear_regression_suite
from tree_models import run_tree_models_suite
from unsupervised_learning import run_unsupervised_suite
from model_evaluation import run_evaluation_suite
from ml_engineering import FeatureStore, ProductionPredictor, assess_data_drift, PACKAGE_DIR
from lifecycle_tracer import trace_prediction_request, get_full_lifecycle_overview

app = Flask(__name__)

# Global singletons / caches
_cache = {}
predictor = None

def get_cached_or_compute(key, compute_fn):
    if key not in _cache:
        _cache[key] = compute_fn()
    return _cache[key]

def get_production_predictor():
    global predictor
    if predictor is None:
        predictor = ProductionPredictor()
    return predictor


# =====================================================================
# UI ROUTES
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html", active="none")


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = 0
    try:
        summary = get_data_summary()
        duplicate_count = get_duplicate_count()
    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "data_loading.html",
        active="data-loading",
        summary=summary,
        Duplicate_count=duplicate_count,
        error=error
    )


@app.route("/preprocessing")
def preprocessing_route():
    error = None
    summary = None
    try:
        summary = run_preprocessing()
    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        summary=summary,
        error=error
    )


@app.route("/eda")
def eda_route():
    error = None
    summary = None
    try:
        summary = get_eda_summary_data()
    except Exception as e:
        error = f"Error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        summary=summary,
        error=error
    )


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression_route():
    error = None
    summary = None
    predicted_salary = None
    cgpa_input = None

    if request.method == "POST":
        try:
            cgpa_input = float(request.form.get("cgpa", 0))
            predicted_salary = round(predict_sample_salary(cgpa_input, model_type="simple"), 2)
        except Exception as e:
            error = f"Prediction error: {e}"

    try:
        summary = get_linear_regression_summary()
    except Exception as e:
        if not error:
            error = f"Model execution error: {e}"

    return render_template(
        "linear_regression.html",
        active="linear-regression",
        summary=summary,
        predicted_salary=predicted_salary,
        cgpa_input=cgpa_input,
        error=error
    )


@app.route("/linear-models")
def linear_models_route():
    error = None
    summary = None
    try:
        summary = get_cached_or_compute(
            "linear_models",
            lambda: run_linear_regression_suite(sample_size=10000, save_plots=False)
        )
    except Exception as e:
        error = f"Linear models error: {e}"

    return render_template(
        "linear_models.html",
        active="linear-models",
        summary=summary,
        error=error
    )


@app.route("/tree-models")
def tree_models_route():
    error = None
    summary = None
    try:
        summary = get_cached_or_compute(
            "tree_models",
            lambda: run_tree_models_suite(sample_size=10000, save_plots=False)
        )
    except Exception as e:
        error = f"Tree models error: {e}"

    return render_template(
        "tree_models.html",
        active="tree-models",
        summary=summary,
        error=error
    )


@app.route("/unsupervised")
def unsupervised_route():
    error = None
    summary = None
    try:
        summary = get_cached_or_compute(
            "unsupervised",
            lambda: run_unsupervised_suite(sample_size=3000, save_plots=False)
        )
    except Exception as e:
        error = f"Unsupervised error: {e}"

    return render_template(
        "unsupervised.html",
        active="unsupervised",
        summary=summary,
        error=error
    )


@app.route("/evaluation")
def evaluation_route():
    error = None
    summary = None
    try:
        summary = get_cached_or_compute(
            "evaluation",
            lambda: run_evaluation_suite(sample_size=6000, save_plots=False)
        )
    except Exception as e:
        error = f"Evaluation error: {e}"

    return render_template(
        "evaluation.html",
        active="evaluation",
        summary=summary,
        error=error
    )


@app.route("/ml-engineering")
def ml_engineering_route():
    error = None
    drift_report = None
    try:
        fs = FeatureStore()
        # Assess drift on a recent slice
        sample_batch = fs.get_historical_features(sample_size=500)
        drift_report = assess_data_drift(sample_batch)
    except Exception as e:
        error = f"ML engineering error: {e}"

    return render_template(
        "ml_engineering.html",
        active="ml-engineering",
        drift_report=drift_report,
        error=error
    )


@app.route("/lifecycle-tracer", methods=["GET", "POST"])
def lifecycle_tracer_route():
    error = None
    trace = None
    req_input = None

    if request.method == "POST":
        try:
            req_input = {
                "Gender": "Male", "City": "Bangalore", "CollegeTier": 1, "Stream": "Computer Science",
                "Specialisation": "Data Science", "Hostel": "No", "HistoryOfBacklogs": "No",
                "CGPA_Tier": "High", "SGPA_Sem1": float(request.form.get("CGPA", 8.5)),
                "SGPA_Sem2": float(request.form.get("CGPA", 8.5)), "SGPA_Sem3": float(request.form.get("CGPA", 8.5)),
                "SGPA_Sem4": float(request.form.get("CGPA", 8.5)), "SGPA_Sem5": float(request.form.get("CGPA", 8.5)),
                "SGPA_Sem6": float(request.form.get("CGPA", 8.5)), "SGPA_Sem7": float(request.form.get("CGPA", 8.5)),
                "SGPA_Sem8": float(request.form.get("CGPA", 8.5)), "CGPA": float(request.form.get("CGPA", 8.5)),
                "AttendancePercent": 88.0, "Internships": int(request.form.get("Internships", 2)),
                "Projects": 3, "Workshops": 2, "Certifications": 2, "Publications": 0,
                "AptitudeTestScore": float(request.form.get("AptitudeTestScore", 85)),
                "SoftSkillsRating": 4.5, "CodingTestScore": float(request.form.get("CodingTestScore", 90)),
                "MockInterviewScore": 88.0
            }
            trace = trace_prediction_request(req_input)
        except Exception as e:
            error = f"Trace error: {e}"
    else:
        # Default trace sample on first load
        trace = trace_prediction_request()

    lifecycle_overview = get_full_lifecycle_overview()

    return render_template(
        "lifecycle_tracer.html",
        active="lifecycle-tracer",
        trace=trace,
        request_input=req_input,
        lifecycle_overview=lifecycle_overview,
        error=error
    )


@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    error = None
    result = None
    latency = None
    form_data = None
    prod_pred = get_production_predictor()

    if request.method == "POST":
        try:
            form_data = request.form.to_dict()
            # Construct feature dictionary with appropriate type casts
            payload = {
                "Gender": form_data.get("Gender", "Male"),
                "City": form_data.get("City", "Bangalore"),
                "CollegeTier": int(form_data.get("CollegeTier", 1)),
                "Stream": form_data.get("Stream", "Computer Science"),
                "Specialisation": form_data.get("Specialisation", "Data Science"),
                "Hostel": form_data.get("Hostel", "No"),
                "HistoryOfBacklogs": form_data.get("HistoryOfBacklogs", "No"),
                "CGPA_Tier": form_data.get("CGPA_Tier", "High"),
                "SGPA_Sem1": float(form_data.get("SGPA_Sem1", form_data.get("CGPA", 8.5))),
                "SGPA_Sem2": float(form_data.get("SGPA_Sem2", form_data.get("CGPA", 8.5))),
                "SGPA_Sem3": float(form_data.get("SGPA_Sem3", form_data.get("CGPA", 8.5))),
                "SGPA_Sem4": float(form_data.get("SGPA_Sem4", form_data.get("CGPA", 8.5))),
                "SGPA_Sem5": float(form_data.get("SGPA_Sem5", form_data.get("CGPA", 8.5))),
                "SGPA_Sem6": float(form_data.get("SGPA_Sem6", form_data.get("CGPA", 8.5))),
                "SGPA_Sem7": float(form_data.get("SGPA_Sem7", form_data.get("CGPA", 8.5))),
                "SGPA_Sem8": float(form_data.get("SGPA_Sem8", form_data.get("CGPA", 8.5))),
                "CGPA": float(form_data.get("CGPA", 8.5)),
                "AttendancePercent": float(form_data.get("AttendancePercent", 85.0)),
                "Internships": int(form_data.get("Internships", 2)),
                "Projects": int(form_data.get("Projects", 3)),
                "Workshops": int(form_data.get("Workshops", 2)),
                "Certifications": int(form_data.get("Certifications", 2)),
                "Publications": int(form_data.get("Publications", 0)),
                "AptitudeTestScore": float(form_data.get("AptitudeTestScore", 80.0)),
                "SoftSkillsRating": float(form_data.get("SoftSkillsRating", 4.0)),
                "CodingTestScore": float(form_data.get("CodingTestScore", 85.0)),
                "MockInterviewScore": float(form_data.get("MockInterviewScore", 85.0))
            }

            resp = prod_pred.predict_placement(payload)
            result = resp["results"]
            latency = resp["latency_ms"]
        except Exception as e:
            error = f"Inference error: {e}"

    return render_template(
        "predict.html",
        active="predict",
        result=result,
        latency=latency,
        model_version=prod_pred.manifest["version"],
        form_data=form_data,
        error=error
    )


# =====================================================================
# PRODUCTION REST API ENDPOINTS
# =====================================================================

@app.route("/api/v1/health", methods=["GET"])
def api_health():
    prod_pred = get_production_predictor()
    return jsonify({
        "status": "healthy",
        "service": "Placement Prediction Microservice",
        "timestamp": datetime.now().isoformat(),
        "model_version": prod_pred.manifest.get("version", "1.0.0"),
        "model_id": prod_pred.manifest.get("model_id", "unknown")
    }), 200


@app.route("/api/v1/model/metadata", methods=["GET"])
def api_model_metadata():
    prod_pred = get_production_predictor()
    return jsonify(prod_pred.manifest), 200


@app.route("/api/v1/features/<int:student_id>", methods=["GET"])
def api_get_features(student_id):
    fs = FeatureStore()
    feat = fs.get_online_features(student_id)
    if not feat:
        return jsonify({"error": f"StudentID {student_id} not found in feature store"}), 404
    return jsonify({
        "student_id": student_id,
        "features": feat
    }), 200


@app.route("/api/v1/predict/placement", methods=["POST"])
def api_predict_placement():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    prod_pred = get_production_predictor()
    try:
        response = prod_pred.predict_placement(payload)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/predict/salary", methods=["POST"])
def api_predict_salary():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No JSON payload provided"}), 400

    prod_pred = get_production_predictor()
    try:
        if isinstance(payload, dict):
            df_in = pd.DataFrame([payload])
            sal = float(prod_pred.reg.predict(df_in)[0])
            return jsonify({"predicted_salary_lpa": round(max(0.0, sal), 2)}), 200
        else:
            df_in = pd.DataFrame(payload)
            sals = [round(max(0.0, float(s)), 2) for s in prod_pred.reg.predict(df_in)]
            return jsonify({"predicted_salaries_lpa": sals}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/monitoring/drift", methods=["POST"])
def api_monitoring_drift():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error": "No batch payload provided"}), 400

    try:
        df_batch = pd.DataFrame(payload if isinstance(payload, list) else [payload])
        drift_report = assess_data_drift(df_batch)
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "batch_size": len(df_batch),
            "drift_report": drift_report
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)