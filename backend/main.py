from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from backend import config,storage
from backend.benchmark import run
from backend.providers import generate,ProviderError
app=FastAPI(title="LLM Benchmarking & ModelOps MVP",version="0.1.0")
storage.init_db()
class Query(BaseModel):
    question:str
    context:str=""
@app.get("/health")
def health(): return {"status":"ok","provider":config.PROVIDER}
@app.get("/models")
def models(): return {"provider":config.PROVIDER,"model":config.OLLAMA_MODEL if config.PROVIDER=="ollama" else config.OPENAI_MODEL if config.PROVIDER=="openai_compatible" else "mock-baseline"}
@app.post("/query")
def query(q:Query):
    try:
        answer,model=generate(q.question,q.context); return {"model":model,"answer":answer}
    except ProviderError as e: raise HTTPException(502,str(e))
@app.post("/benchmark")
def benchmark(dataset:str="data/benchmark.jsonl"):
    try: return run(dataset)
    except Exception as e: raise HTTPException(400,str(e))
@app.get("/runs")
def runs(): return storage.list_runs()
@app.get("/runs/{rid}")
def details(rid:str):
    r=storage.get_run(rid)
    if not r: raise HTTPException(404,"Run not found")
    return r
@app.get("/metrics")
def metrics():
    rows=storage.list_runs()
    return {"run_count":len(rows),"latest":rows[0] if rows else None}
