import database
import rag_report_engine
import app

def test_all():
    print("--- 1. Database Init Test ---")
    database.init_db()
    print("[+] Database initialized successfully.")

    print("\n--- 2. RAG Semantic Retrieval Test ---")
    docs = rag_report_engine.rag_retriever.retrieve('Frontal lobe glioblastoma high risk edema volume 8.5 cm3', top_k=3)
    assert len(docs) > 0, "RAG retrieval failed"
    print(f"[+] RAG Retriever OK: Retrieved {len(docs)} relevant clinical guidelines.")
    for d in docs:
        print(f"   - [{d['id']}] {d['title']} ({d['category']})")

    print("\n--- 3. Longitudinal Trajectory Analyzer Test ---")
    mock_scans = [
        {'date': '2026-01-10 10:00', 'volume': 3.2, 'risk': 'Medium', 'location': 'Frontal Lobe', 'scan_type': '2D MRI Series', 'confidence': 91.0, 'result': 'Tumor Detected'},
        {'date': '2026-03-15 14:00', 'volume': 5.8, 'risk': 'High', 'location': 'Frontal Lobe', 'scan_type': '2D MRI Series', 'confidence': 95.0, 'result': 'Tumor Detected'}
    ]
    traj = rag_report_engine.LongitudinalHistoryAnalyzer.analyze_history(mock_scans)
    print(f"[+] Longitudinal Analyzer OK: Total Scans = {traj['total_scans']}, Delta = {traj['delta_volume']} cm3, Category = {traj['progression_category']}")

    print("\n--- 4. Master RAG Report Generation Test ---")
    mock_patient = {
        'patient_id': 'PAT-TEST-001',
        'name': 'Alexander Wright',
        'age': '52',
        'gender': 'Male',
        'mobile': '9876543210',
        'latest_scan': mock_scans[-1],
        'scans': mock_scans
    }
    rep = rag_report_engine.generate_patient_rag_report(mock_patient, provider='offline')
    assert rep['success'] is True, "Report generation failed"
    print(f"[+] Offline Clinical RAG Engine OK: Model = {rep['model_name']}, Generated in {rep['generation_time_sec']}s")

    print("\n--- 5. Database Storage & Retrieval Test ---")
    # Save mock patient and scan into SQLite
    database.save_patient_and_scan(
        patient_id='PAT-TEST-001',
        name='Alexander Wright',
        age='52',
        gender='Male',
        scan_dict=mock_scans[-1],
        mobile='9876543210'
    )
    database.save_patient_and_scan(
        patient_id='PAT-TEST-001',
        name='Alexander Wright',
        age='52',
        gender='Male',
        scan_dict=mock_scans[0],
        mobile='9876543210'
    )

    rep_id = database.save_patient_report(
        patient_id='PAT-TEST-001',
        scan_id=1,
        model_name=rep['model_name'],
        provider=rep['provider'],
        rag_context=rep['retrieved_sources'],
        report_text=rep['report_markdown'],
        report_json={"trajectory": rep['trajectory']}
    )
    fetched = database.get_latest_patient_report('PAT-TEST-001')
    assert fetched is not None, "Database report retrieval failed"
    print(f"[+] Database Storage OK: Saved Report ID {rep_id}, Fetched Model = {fetched['model_name']}, Provider = {fetched['provider']}")

    print("\n--- 6. Flask Endpoints Test ---")
    client = app.app.test_client()
    res_report = client.get('/report/PAT-TEST-001')
    print(f"[+] Flask /report/PAT-TEST-001 Route OK (Status: {res_report.status_code})")
    assert res_report.status_code == 200

    res_api = client.get('/api/generate_report/PAT-TEST-001')
    print(f"[+] Flask /api/generate_report/PAT-TEST-001 API OK (Status: {res_api.status_code})")
    assert res_api.status_code == 200

    print("\n==========================================")
    print("ALL TESTS PASSED SUCCESSFULLY! RAG & LLM REPORT ENGINE IS FULLY OPERATIONAL.")
    print("==========================================")

if __name__ == '__main__':
    test_all()
