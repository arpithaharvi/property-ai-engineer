# ============================================================
# modules/database.py — SQLite Persistence Layer
#
# REBUILT — this file was found empty on disk (likely lost during
# a folder copy / venv recreation). Rebuilt to match exactly what
# app.py imports and calls:
#   init_db()
#   save_document(...)
#   save_property_with_analysis(...)
#   save_excel_batch(...)
#   get_all_documents()
#   get_all_excel_batches()
#
# Tables (per original roadmap, Session 5):
#   properties       -- one row per analysed property (PDF mode)
#   documents        -- one row per uploaded/processed PDF file
#   excel_batches     -- one row per Excel batch analysis run
#
# Note: kept simpler than the original 5-table roadmap design
# (owners/schedules/boundaries/extracted_text as separate tables)
# since app.py currently only ever calls save_document and
# save_property_with_analysis with flat dicts — splitting into
# many tables would need matching changes in app.py too. This
# version stores extracted/validation/score data as JSON columns
# inside "properties", which is simpler and fully compatible with
# your current app.py without further changes.
# ============================================================

import json
import os
import sqlite3
from datetime import datetime

# ── Database file location ──
# Uses the same "database/" folder your roadmap specifies
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_PATH = os.path.join(DB_DIR, "properties.db")


def _get_connection():
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates all required tables if they don't already exist.
    Safe to call every time the app starts — CREATE TABLE IF NOT EXISTS
    means existing data is never wiped.
    """
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            total_pages INTEGER,
            digital_pages INTEGER,
            scanned_pages INTEGER,
            languages TEXT,
            case_type TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            analysis_date TEXT NOT NULL,
            property_type TEXT,
            extracted_json TEXT,
            scores_json TEXT,
            validation_json TEXT,
            is_bank_ready INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS excel_batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            total_properties INTEGER,
            results_json TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_document(filename, total_pages, digital_pages, scanned_pages, languages, case_type):
    """
    Saves one processed PDF document's metadata.
    Called from both Upload Documents page and Risk Analysis PDF mode.
    Returns the new document's database ID.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO documents
            (filename, upload_date, total_pages, digital_pages, scanned_pages, languages, case_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        filename,
        datetime.now().isoformat(),
        total_pages,
        digital_pages,
        scanned_pages,
        languages,
        case_type,
    ))
    doc_id = cur.lastrowid
    conn.commit()
    conn.close()
    return doc_id


def save_property_with_analysis(document_id, extracted, scores, validation):
    """
    Saves one property's full analysis: extracted fields, AI scores,
    and validation flags. Stored as JSON columns for flexibility since
    the extracted schema now varies by property type (Residential,
    Flat, Commercial, IT_Park, Vacant_Site, Industrial).
    Returns the new property's database ID.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO properties
            (document_id, analysis_date, property_type, extracted_json,
             scores_json, validation_json, is_bank_ready)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        document_id,
        datetime.now().isoformat(),
        extracted.get("property_type", "Unknown"),
        json.dumps(extracted),
        json.dumps(scores),
        json.dumps(validation),
        1 if validation.get("is_bank_ready") else 0,
    ))
    prop_id = cur.lastrowid
    conn.commit()
    conn.close()
    return prop_id


def save_excel_batch(filename, result_df):
    """
    Saves one Excel batch analysis run. result_df is a pandas DataFrame —
    stored as JSON records for retrieval later.
    Returns the new batch's database ID.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO excel_batches (filename, upload_date, total_properties, results_json)
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        datetime.now().isoformat(),
        len(result_df),
        result_df.to_json(orient="records"),
    ))
    batch_id = cur.lastrowid
    conn.commit()
    conn.close()
    return batch_id


def get_all_documents():
    """
    Returns a list of dicts, one per processed document, for the
    Reports page. Most recent first.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_excel_batches():
    """
    Returns a list of dicts, one per Excel batch run, for the
    Reports page. Most recent first. Note: results_json is left as
    a raw JSON string here (Reports page just displays batch-level
    metadata) — parse it with json.loads() if you need the full
    per-row results back out.
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, filename, upload_date, total_properties FROM excel_batches ORDER BY id DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_property_by_id(property_id):
    """
    Bonus helper (not currently called by app.py, but useful for
    Week 6+ agents that may want to re-load a specific property's
    saved analysis by ID).
    """
    conn = _get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM properties WHERE id = ?", (property_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    result = dict(row)
    result["extracted_json"] = json.loads(result["extracted_json"])
    result["scores_json"] = json.loads(result["scores_json"])
    result["validation_json"] = json.loads(result["validation_json"])
    return result