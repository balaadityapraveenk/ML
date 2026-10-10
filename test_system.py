import json
from app import app

def test_all():
    client = app.test_client()
    print("Testing Web and API Endpoints...")

    # Test Web Pages
    routes = [
        "/",
        "/data-loading",
        "/preprocessing",
        "/eda",
        "/linear-models",
        "/tree-models",
        "/unsupervised",
        "/evaluation",
        "/ml-engineering",
        "/lifecycle-tracer",
        "/predict"
    ]

    for r in routes:
        resp = client.get(r)
        print(f"GET {r:22s} -> Status: {resp.status_code}")
        assert resp.status_code == 200, f"Route {r} failed with status {resp.status_code}"

    # Test REST APIs
    # 1. Health
    resp = client.get("/api/v1/health")
    print(f"GET /api/v1/health          -> Status: {resp.status_code}, Body: {resp.get_json()['status']}")
    assert resp.status_code == 200

    # 2. Metadata
    resp = client.get("/api/v1/model/metadata")
    print(f"GET /api/v1/model/metadata  -> Status: {resp.status_code}, Model: {resp.get_json()['model_id']}")
    assert resp.status_code == 200

    # 3. Online Features
    resp = client.get("/api/v1/features/1")
    print(f"GET /api/v1/features/1      -> Status: {resp.status_code}, Found: {'features' in resp.get_json()}")
    assert resp.status_code == 200

    # 4. Placement Prediction
    sample = {
        "Gender": "Male", "City": "Bangalore", "CollegeTier": 1, "Stream": "Computer Science",
        "Specialisation": "Data Science", "Hostel": "No", "HistoryOfBacklogs": "No",
        "CGPA_Tier": "High", "SGPA_Sem1": 8.5, "SGPA_Sem2": 8.6, "SGPA_Sem3": 8.8,
        "SGPA_Sem4": 8.7, "SGPA_Sem5": 9.0, "SGPA_Sem6": 9.1, "SGPA_Sem7": 8.9,
        "SGPA_Sem8": 9.2, "CGPA": 8.9, "AttendancePercent": 92.0, "Internships": 2,
        "Projects": 4, "Workshops": 3, "Certifications": 2, "Publications": 1,
        "AptitudeTestScore": 88.0, "SoftSkillsRating": 4.5, "CodingTestScore": 95.0,
        "MockInterviewScore": 90.0
    }
    resp = client.post("/api/v1/predict/placement", json=sample)
    pred_data = resp.get_json()
    print(f"POST /api/v1/predict/placement -> Status: {resp.status_code}, Status: {pred_data['results']['placement_status']}, Prob: {pred_data['results']['placement_probability']}")
    assert resp.status_code == 200

    # 5. Salary Prediction
    resp = client.post("/api/v1/predict/salary", json=sample)
    print(f"POST /api/v1/predict/salary -> Status: {resp.status_code}, Salary: {resp.get_json()['predicted_salary_lpa']} LPA")
    assert resp.status_code == 200

    # 6. Drift Monitoring
    resp = client.post("/api/v1/monitoring/drift", json=[sample])
    print(f"POST /api/v1/monitoring/drift -> Status: {resp.status_code}, Monitored: {list(resp.get_json()['drift_report'].keys())}")
    assert resp.status_code == 200

    print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    test_all()
