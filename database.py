import sqlite3
import json
from datetime import datetime

DB_NAME = "patients.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Master Patients Table (with Mobile number)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age TEXT,
            gender TEXT,
            mobile TEXT DEFAULT '',
            created_at TEXT
        )
    """)

    # 2. Individual Scans Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            scan_type TEXT DEFAULT '2D MRI',
            result TEXT,
            tumor_type TEXT,
            volume REAL DEFAULT 0.0,
            risk TEXT DEFAULT 'Low',
            confidence REAL DEFAULT 0.0,
            plot_file TEXT,
            slice_images TEXT,
            date TEXT,
            location TEXT DEFAULT '',
            pathology TEXT DEFAULT '',
            recommendation TEXT DEFAULT '',
            et_volume REAL DEFAULT 0.0,
            ed_volume REAL DEFAULT 0.0,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    """)

    # Add missing columns dynamically to patients table
    cursor.execute("PRAGMA table_info(patients)")
    existing_patient_cols = [row['name'] for row in cursor.fetchall()]
    if 'mobile' not in existing_patient_cols:
        try:
            cursor.execute("ALTER TABLE patients ADD COLUMN mobile TEXT DEFAULT ''")
        except Exception:
            pass

    # Add missing columns dynamically to scans table
    cursor.execute("PRAGMA table_info(scans)")
    existing_scan_cols = [row['name'] for row in cursor.fetchall()]
    for col, col_type in [
        ('location', "TEXT DEFAULT ''"),
        ('pathology', "TEXT DEFAULT ''"),
        ('recommendation', "TEXT DEFAULT ''"),
        ('et_volume', "REAL DEFAULT 0.0"),
        ('ed_volume', "REAL DEFAULT 0.0")
    ]:
        if col not in existing_scan_cols:
            try:
                cursor.execute(f"ALTER TABLE scans ADD COLUMN {col} {col_type}")
            except Exception:
                pass

    # 3. AI RAG & LLM Generated Reports Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            scan_id INTEGER,
            model_name TEXT,
            provider TEXT,
            rag_context TEXT,
            report_text TEXT,
            report_json TEXT,
            created_at TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    """)

    conn.commit()
    conn.close()

def save_patient_and_scan(patient_id, name, age, gender, scan_dict, mobile=""):
    """Inserts or updates patient details and logs a new scan for this patient."""
    conn = get_db_connection()
    cursor = conn.cursor()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    if not mobile:
        mobile = scan_dict.get('mobile', '')
    
    # 1. Upsert patient
    cursor.execute("SELECT id, mobile FROM patients WHERE patient_id = ? OR (mobile != '' AND mobile = ?)", (patient_id, mobile))
    existing = cursor.fetchone()
    if existing:
        cursor.execute("""
            UPDATE patients SET name = ?, age = ?, gender = ?, mobile = COALESCE(NULLIF(?, ''), mobile)
            WHERE patient_id = ?
        """, (name, age, gender, mobile, patient_id))
    else:
        cursor.execute("""
            INSERT INTO patients (patient_id, name, age, gender, mobile, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, name, age, gender, mobile, now_str))

    # 2. Insert scan record with medical biomarkers
    slice_imgs_json = json.dumps(scan_dict.get('slice_images', []))
    cursor.execute("""
        INSERT INTO scans (
            patient_id, scan_type, result, tumor_type, volume, risk, confidence, plot_file, slice_images, date,
            location, pathology, recommendation, et_volume, ed_volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_id,
        scan_dict.get('scan_type', '2D MRI Series'),
        scan_dict.get('result', 'No Tumor Detected'),
        scan_dict.get('tumor_type', 'None'),
        float(scan_dict.get('volume', 0.0)),
        scan_dict.get('risk', 'Low'),
        float(scan_dict.get('confidence', 0.0)),
        scan_dict.get('plot_file', ''),
        slice_imgs_json,
        scan_dict.get('date', now_str),
        scan_dict.get('location', ''),
        scan_dict.get('pathology', ''),
        scan_dict.get('recommendation', ''),
        float(scan_dict.get('et_volume', 0.0)),
        float(scan_dict.get('ed_volume', 0.0))
    ))

    scan_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return scan_id

def insert_patient(patient_data):
    """Legacy helper for backward compatibility."""
    if len(patient_data) >= 8:
        pid, name, age, gender, volume, risk, confidence, date = patient_data[:8]
        save_patient_and_scan(pid, name, age, gender, {
            'volume': volume,
            'risk': risk,
            'confidence': confidence,
            'date': date,
            'result': "Tumor Detected" if float(volume or 0) > 0 else "No Tumor Detected",
            'tumor_type': "Positive" if float(volume or 0) > 0 else "None"
        })

def get_patient(identifier):
    """Fetches patient info by Patient ID or Mobile along with their latest scan summary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_id = str(identifier).strip()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ? OR mobile = ?", (clean_id, clean_id))
    p = cursor.fetchone()
    if not p:
        conn.close()
        return None

    pid = p['patient_id']
    cursor.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY id DESC LIMIT 1", (pid,))
    s = cursor.fetchone()
    conn.close()

    p_dict = dict(p)
    if s:
        p_dict.update({
            'latest_scan_id': s['id'],
            'volume': s['volume'],
            'risk': s['risk'],
            'confidence': s['confidence'],
            'date': s['date'],
            'scan_type': s['scan_type'],
            'result': s['result'],
            'tumor_type': s['tumor_type'],
            'plot_file': s['plot_file']
        })
        legacy_tuple = (
            p['id'], p['patient_id'], p['name'], p['age'], p['gender'],
            s['volume'], s['risk'], s['confidence'], s['date']
        )
    else:
        legacy_tuple = (
            p['id'], p['patient_id'], p['name'], p['age'], p['gender'],
            0.0, 'Low', 0.0, p['created_at']
        )
    return legacy_tuple

def get_patient_dict(identifier):
    """Returns clean patient dictionary with full scan list by Patient ID or Mobile."""
    if not identifier:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    clean_id = str(identifier).strip()

    # Match by patient_id, mobile, or partial PAT-<mobile>
    cursor.execute("""
        SELECT * FROM patients 
        WHERE patient_id = ? OR mobile = ? OR patient_id = ?
    """, (clean_id, clean_id, f"PAT-{clean_id}"))
    p = cursor.fetchone()
    if not p:
        conn.close()
        return None

    p_dict = dict(p)
    pid = p_dict['patient_id']
    cursor.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY id DESC", (pid,))
    scans = [dict(row) for row in cursor.fetchall()]
    conn.close()

    for sc in scans:
        try:
            sc['slice_images'] = json.loads(sc['slice_images']) if sc.get('slice_images') else []
        except Exception:
            sc['slice_images'] = []

    p_dict['scans'] = scans
    p_dict['total_scans'] = len(scans)
    if scans:
        p_dict['latest_scan'] = scans[0]
        p_dict['volume'] = scans[0]['volume']
        p_dict['risk'] = scans[0]['risk']
        p_dict['confidence'] = scans[0]['confidence']
        p_dict['date'] = scans[0]['date']
        p_dict['result'] = scans[0]['result']
        p_dict['tumor_type'] = scans[0]['tumor_type']
        p_dict['plot_file'] = scans[0]['plot_file']
        p_dict['location'] = scans[0].get('location', '')
        p_dict['pathology'] = scans[0].get('pathology', '')
        p_dict['recommendation'] = scans[0].get('recommendation', '')
        p_dict['et_volume'] = scans[0].get('et_volume', 0.0)
        p_dict['ed_volume'] = scans[0].get('ed_volume', 0.0)
    return p_dict

def get_patient_scans(patient_id):
    """Returns all scans for a specific patient in chronological order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY id ASC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    scans = []
    for r in rows:
        d = dict(r)
        try:
            d['slice_images'] = json.loads(d['slice_images']) if d.get('slice_images') else []
        except Exception:
            d['slice_images'] = []
        scans.append(d)
    return scans

def get_patient_growth(patient_id_or_mobile):
    """Extracts tumor volume progression across all scans for a patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    q = str(patient_id_or_mobile).strip()
    cursor.execute("""
        SELECT s.volume FROM scans s
        JOIN patients p ON s.patient_id = p.patient_id
        WHERE p.patient_id = ? OR p.mobile = ? OR p.name = ?
        ORDER BY s.id ASC
    """, (q, q, q))
    rows = cursor.fetchall()
    conn.close()

    volumes = [float(r['volume'] or 0) for r in rows]
    return volumes if volumes else [0]

def get_patient_history(name):
    """Legacy helper returning history by name."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.id, p.patient_id, p.name, p.age, p.gender, s.volume, s.risk, s.confidence, s.date
        FROM scans s
        JOIN patients p ON s.patient_id = p.patient_id
        WHERE p.name = ? ORDER BY s.id ASC
    """, (name,))
    rows = cursor.fetchall()
    conn.close()
    return [tuple(r) for r in rows]

def get_all_patients():
    """Returns all scan events as flat tuples for legacy table compatibility."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, p.patient_id, p.name, p.age, p.gender, s.volume, s.risk, s.confidence, s.date, s.scan_type
        FROM scans s
        JOIN patients p ON s.patient_id = p.patient_id
        ORDER BY s.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [tuple(r) for r in rows]

def get_all_patients_summary():
    """Returns a consolidated list of distinct patients with their aggregate scan stats."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.patient_id, p.name, p.age, p.gender, p.mobile, p.created_at,
            COUNT(s.id) as scan_count,
            MAX(s.date) as latest_scan_date,
            (SELECT s2.volume FROM scans s2 WHERE s2.patient_id = p.patient_id ORDER BY s2.id DESC LIMIT 1) as latest_volume,
            (SELECT s2.risk FROM scans s2 WHERE s2.patient_id = p.patient_id ORDER BY s2.id DESC LIMIT 1) as latest_risk,
            (SELECT s2.confidence FROM scans s2 WHERE s2.patient_id = p.patient_id ORDER BY s2.id DESC LIMIT 1) as latest_confidence,
            (SELECT s2.scan_type FROM scans s2 WHERE s2.patient_id = p.patient_id ORDER BY s2.id DESC LIMIT 1) as latest_scan_type
        FROM patients p
        LEFT JOIN scans s ON p.patient_id = s.patient_id
        GROUP BY p.patient_id
        ORDER BY p.id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def search_patients(query=""):
    """Quick lookup for auto-completing Patient IDs, names, and mobile numbers."""
    conn = get_db_connection()
    cursor = conn.cursor()
    q = f"%{query.strip()}%"
    cursor.execute("""
        SELECT p.patient_id, p.name, p.age, p.gender, p.mobile, COUNT(s.id) as scan_count, MAX(s.date) as last_date
        FROM patients p
        LEFT JOIN scans s ON p.patient_id = s.patient_id
        WHERE p.patient_id LIKE ? OR p.name LIKE ? OR p.mobile LIKE ?
        GROUP BY p.patient_id
        ORDER BY p.id DESC
        LIMIT 10
    """, (q, q, q))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def save_patient_report(patient_id, scan_id, model_name, provider, rag_context, report_text, report_json=None):
    """Saves a generated AI RAG medical report into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rag_ctx_str = json.dumps(rag_context) if isinstance(rag_context, (list, dict)) else str(rag_context or "")
    rep_json_str = json.dumps(report_json) if isinstance(report_json, (list, dict)) else str(report_json or "")
    
    cursor.execute("""
        INSERT INTO reports (patient_id, scan_id, model_name, provider, rag_context, report_text, report_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, scan_id, model_name, provider, rag_ctx_str, report_text, rep_json_str, now_str))
    
    rep_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return rep_id

def get_latest_patient_report(patient_id):
    """Retrieves the most recent AI RAG generated report for a patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM reports WHERE patient_id = ? ORDER BY id DESC LIMIT 1
    """, (patient_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    try:
        d['rag_context'] = json.loads(d['rag_context']) if d.get('rag_context') else []
    except Exception:
        pass
    try:
        d['report_json'] = json.loads(d['report_json']) if d.get('report_json') else {}
    except Exception:
        pass
    return d

def get_patient_reports(patient_id):
    """Retrieves all past AI RAG generated reports for a patient."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM reports WHERE patient_id = ? ORDER BY id DESC
    """, (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    reports = []
    for r in rows:
        d = dict(r)
        try:
            d['rag_context'] = json.loads(d['rag_context']) if d.get('rag_context') else []
        except Exception:
            pass
        try:
            d['report_json'] = json.loads(d['report_json']) if d.get('report_json') else {}
        except Exception:
            pass
        reports.append(d)
    return reports