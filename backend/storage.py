import json,sqlite3,uuid
from datetime import datetime,timezone
from backend import config
def connect():
    c=sqlite3.connect(config.DATABASE_PATH); c.row_factory=sqlite3.Row; return c
def init_db():
    with connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT, provider TEXT, model TEXT, dataset_path TEXT, total_items INTEGER, average_latency REAL, average_reference_overlap REAL, passed_gate INTEGER, results_json TEXT)")
def save(data):
    rid=str(uuid.uuid4())
    with connect() as c:
        c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",(rid,datetime.now(timezone.utc).isoformat(),data["provider"],data["model"],data["dataset_path"],data["total_items"],data["average_latency"],data["average_reference_overlap"],int(data["passed_gate"]),json.dumps(data["results"])))
    return rid
def list_runs():
    with connect() as c:
        rows=c.execute("SELECT run_id,created_at,provider,model,total_items,average_latency,average_reference_overlap,passed_gate FROM runs ORDER BY created_at DESC").fetchall()
    return [dict(x) for x in rows]
def get_run(rid):
    with connect() as c: row=c.execute("SELECT * FROM runs WHERE run_id=?",(rid,)).fetchone()
    if not row: return None
    d=dict(row); d["results"]=json.loads(d.pop("results_json")); return d
