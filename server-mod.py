import os
# server.py
import asyncio
import uuid
from fastmcp import FastMCP
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import docker
import io
import tarfile
import base64
from pathlib import Path
import importlib.util
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "backend.py"
_spec = importlib.util.spec_from_file_location("tts_text_to_sql", PIPELINE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load pipeline module from {PIPELINE_PATH}")
_pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _pipeline
_spec.loader.exec_module(_pipeline)

run_direct = getattr(_pipeline, "run_direct")
TABLES_JSON_PATH = Path(os.environ.get("TABLES_JSON_PATH") or (getattr(_pipeline, "TABLES_JSON_PATH")))
UI_TEMPLATE_PATH = Path(__file__).parent / "templates" / "context.html"

app = FastMCP(name="demo-echo", version="0.0.1")
contexts: Dict[str, List[Dict[str, str]]] = {}
sql_exec_num: Dict[str, int] = {}
db_ids: Dict[str, str] = {}
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()
_background_tasks: set[asyncio.Task] = set()

def _merged_context(user_id: str = "global") -> str:
    entries = contexts.get(user_id, [])
    if not entries:
        return ""
    merged: List[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            value = entry.get("context")
            if isinstance(value, str) and value:
                merged.append(value)
            elif value:
                merged.append(str(value))
        elif entry is not None:
            merged.append(str(entry))
    return "\n\n".join(merged)

def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _track_task(task: asyncio.Task) -> None:
    _background_tasks.add(task)

    def _remove_task(_task: asyncio.Task) -> None:
        _background_tasks.discard(_task)

    task.add_done_callback(_remove_task)


async def _set_job_fields(job_id: str, **fields: Any) -> None:
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        progress_entries: List[Dict[str, str]] = job.setdefault("progress", [])
        add_progress: Optional[str] = fields.pop("progress_message", None)
        if add_progress:
            progress_entries.append(
                {
                    "message": add_progress,
                    "timestamp": _utc_now(),
                }
            )
        for key, value in fields.items():
            job[key] = value
        job["updated_at"] = _utc_now()


async def _background_text_to_sql(job_id: str, question: str, user_id: str) -> None:
    await _set_job_fields(job_id, status="running", progress_message="SQL generation started.")
    try:
        db_id = db_ids.get(user_id, "EXAMPLE_DATABASE")
        merged_context = _merged_context(user_id)
        await _set_job_fields(job_id, progress_message=f"Using DB `{db_id}`.")
        sql = await asyncio.to_thread(run_direct, question, merged_context, db_id)
    except Exception as exc:
        await _set_job_fields(
            job_id,
            status="failed",
            error=str(exc),
            progress_message="SQL generation failed.",
        )
        print(f"[text_to_sql job {job_id}] run_direct failed: {exc!r}", flush=True)
        return

    await _set_job_fields(
        job_id,
        status="succeeded",
        result=sql,
        progress_message="SQL generation completed.",
    )


@app.tool(description="Execute the python code and return the output.")
def execute_python(code: str, user_id: str) -> str:
    client = docker.from_env()
    container = client.containers.get(user_id)
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        data = code.encode()
        info = tarfile.TarInfo(name="temporary.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)
    container.put_archive("/home", tar_stream.getvalue())
    exit_code, output = container.exec_run("python /home/temporary.py")
    cmd = ["rm", "-f", "/home/temporary.py"]
    container.exec_run(cmd)
    return output

@app.tool(description="Execute the python code(Matplotlib) and return the Base64 encoded image.")
def plot(code: str, user_id: str) -> str:
    client = docker.from_env()
    print(code)
    try: 
        container = client.containers.get(user_id)
        cmd = ["rm", "-f", "/home/temporary.py"]
        container.exec_run(cmd)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            data = code.encode()
            info = tarfile.TarInfo(name="temporary.py")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)
        container.put_archive("/home", tar_stream.getvalue())
        exit_code, output = container.exec_run("python /home/temporary.py")
        return "success"
    except Exception as e:
         return f"failed by {e}"
    """
    try: stream, stat = container.get_archive("/tmp/output.png")
    except: exit()
    file_like = io.BytesIO(b"".join(stream))
    with tarfile.open(fileobj=file_like) as tar:
        member = tar.getmembers()[0]
        file_data = tar.extractfile(member).read()
    b64_encoded = base64.b64encode(file_data).decode("utf-8")
    b64_encoded = f"data:image/png;base64,{b64_encoded}"
    return b64_encoded
    """

@app.tool(description="Start asynchronous text-to-SQL job. Returns a job_id for polling.")
async def text_to_sql(ctx: Any, question: str, user_id: str = "global") -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    created_at = _utc_now()
    job_payload = {
        "job_id": job_id,
        "status": "queued",
        "question": question,
        "user_id": user_id,
        "result": None,
        "error": None,
        "progress": [],
        "created_at": created_at,
        "updated_at": created_at,
    }
    async with _jobs_lock:
        _jobs[job_id] = job_payload
    progress_cb = getattr(ctx, "progress", None)
    if progress_cb is not None:
        try:
            await progress_cb("text_to_sql job accepted. Poll check_text_to_sql for updates.")
        except Exception:
            pass
    task = asyncio.create_task(_background_text_to_sql(job_id, question, user_id))
    _track_task(task)
    return {
        "job_id": job_id,
        "status": "queued",
        "poll_tool": "check_text_to_sql",
        "suggested_poll_interval": 5,
    }


@app.tool(description="Check the status of a text_to_sql job by job_id.")
async def check_text_to_sql(job_id: str) -> Dict[str, Any]:
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return {
                "job_id": job_id,
                "status": "not_found",
                "error": f"Job {job_id} not found.",
            }
        # Return a shallow copy so callers cannot mutate internal state.
        job_copy = dict(job)
        job_copy["progress"] = list(job.get("progress", []))
        return job_copy
"""
@app.tool(description="Check if analysis and plot is available or not.")
def check() -> bool:
    try:
        client = docker.from_env()
        container = client.containers.get("plot_container")
        bash = "" 
        if [ -f "/tmp/output.csv" ]; then
            echo "file exist"
        else
            echo "no file"
        fi
        ""
        exit_code, output = container.exec_run(cmd)
        if output == "file exist":
            return True
        else: return False
    except: return False
"""

mcp_app = app.http_app(path="/", stateless_http=True)
api = FastAPI(lifespan=mcp_app.lifespan)
api.mount("/mcp", mcp_app)


def _read_ui_template() -> str:
    try:
        return UI_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "<h1>context.html not found</h1>"


@api.get("/ui", response_class=HTMLResponse)
async def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=_read_ui_template())


@api.get("/ui/context")
async def read_context(user_id: str = "global") -> JSONResponse:
    entries = contexts.get(user_id, [])
    latest_entry: Optional[Dict[str, str]] = entries[-1] if entries else None
    latest_context = ""
    if isinstance(latest_entry, dict):
        latest_context = latest_entry.get("context", "") or ""
    elif latest_entry is not None:
        latest_context = str(latest_entry)
    return JSONResponse(
        {
            "context": latest_context,
            "context_entry": latest_entry,
            "history": entries,
            "user_id": user_id,
            "known_users": list(contexts.keys()),
        },
    )


@api.get("/ui/db")
async def read_db_selection(user_id: Optional[str] = None) -> JSONResponse:
    if user_id:
        return JSONResponse(
            {
                "user_id": user_id,
                "db_name": db_ids.get(user_id, ""),
                "db_ids": db_ids,
            },
        )
    return JSONResponse({"db_ids": db_ids})


@api.post("/ui/db")
async def update_db_selection(payload: dict = Body(...)) -> JSONResponse:
    user_id_raw = payload.get("user_id") or "global"
    db_name_raw = payload.get("db_name", payload.get("db"))
    user_id = str(user_id_raw).strip() if user_id_raw is not None else "global"
    if db_name_raw is None:
        db_name = ""
    elif isinstance(db_name_raw, str):
        db_name = db_name_raw.strip()
    else:
        db_name = str(db_name_raw)
    db_ids[user_id] = db_name
    return JSONResponse(
        {
            "user_id": user_id,
            "db_name": db_name,
            "db_ids": db_ids,
        },
    )


@api.post("/ui/sql_exec")
async def increment_sql_exec(payload: dict = Body(...)) -> JSONResponse:
    user_id_raw = payload.get("user_id") if isinstance(payload, dict) else None
    user_id = str(user_id_raw).strip() if user_id_raw is not None else "global"
    if not user_id:
        user_id = "global"
    count = sql_exec_num.get(user_id, 0) + 1
    sql_exec_num[user_id] = count
    return JSONResponse({
        "user_id": user_id,
        "count": count,
        "sql_exec_num": sql_exec_num,
    })


@api.get("/ui/sql_exec")
async def read_sql_exec(user_id: Optional[str] = None) -> JSONResponse:
    if user_id is None or not str(user_id).strip():
        return JSONResponse({
            "sql_exec_num": sql_exec_num,
        })
    key = str(user_id).strip()
    return JSONResponse({
        "user_id": key,
        "count": sql_exec_num.get(key, 0),
        "sql_exec_num": sql_exec_num,
    })


@api.post("/ui/context")
async def update_context(payload: dict = Body(...)) -> JSONResponse:
    user_id = payload.get("user_id") or "global"
    context_value = payload.get("context", payload.get("text", ""))
    if context_value is None:
        context_value = ""
    if not isinstance(context_value, str):
        context_value = str(context_value)
    created_at_raw = payload.get("created_at", "")
    created_at: str
    if isinstance(created_at_raw, str) and created_at_raw.strip():
        created_at = created_at_raw
    else:
        created_at = datetime.utcnow().isoformat() + "Z"
    entry = {
        "context": context_value,
        "created_at": created_at,
    }
    bucket = contexts.setdefault(user_id, [])
    bucket.append(entry)
    if len(bucket) > 200:
        del bucket[:-200]
    return JSONResponse(
        {
            "context": context_value,
            "context_entry": entry,
            "history": bucket,
            "user_id": user_id,
            "known_users": list(contexts.keys()),
        },
    )


@api.get("/jobs/{job_id}")
async def read_job(job_id: str) -> JSONResponse:
    async with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
        job_copy = dict(job)
        job_copy["progress"] = list(job.get("progress", []))
    return JSONResponse(job_copy)


@api.get("/jobs")
async def list_jobs() -> JSONResponse:
    async with _jobs_lock:
        items = []
        for job_id, payload in _jobs.items():
            job_copy = dict(payload)
            job_copy["progress"] = list(payload.get("progress", []))
            job_copy["job_id"] = job_id
            items.append(job_copy)
    return JSONResponse({"jobs": items})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="127.0.0.1", port = 1234)
