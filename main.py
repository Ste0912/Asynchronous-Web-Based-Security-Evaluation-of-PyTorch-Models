import time
import os
import json
import redis
from fastapi import Query
from typing import Dict, Any
from typing import Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from celery.result import AsyncResult
from worker import create_task


app = FastAPI()
# --- Redis metadata store (separate DB from Celery backend) ---
APP_REDIS_URL = os.getenv("APP_REDIS_URL", "redis://localhost:6379/1")
r = redis.Redis.from_url(APP_REDIS_URL, decode_responses=True)

JOB_KEY_PREFIX = "secml:job:"        # hash per job: secml:job:<job_id>
JOBS_INDEX_KEY = "secml:jobs:index"  # sorted set of job_ids by submitted_at


# --- Serve the "static" folder at the "/static" URL ---
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


#Redirect root URL (/) to the dashboard
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# POST endpoint to submit a Model Name AND Attack Parameters
@app.post("/submit_eval")
def submit_eval(
        model_name: str,
        attack_type: str = "pgd-linf",
        epsilon: float = 0.03137,  # Default to 8/255
        num_steps: int = 10,       # Default to 10
        step_size: float = 0.00784, # Default to 2/255
        gamma: float = 0.05,        # FMN Default to 0.05
        perturbation_model: str = "linf",
        # epsilon sweep params (used by FMN curve; harmless for PGD)
        eps_min: float = 0.0,
        eps_max: float = 0.1,
        eps_points: int = 21,
        target: Optional[int] = None,
                ):
   
    #Capture the timestamp of submission (Enqueue Time)
    submit_time = time.time()
    # Send task to Redis, passing ALL parameters to the worker
    task = create_task.delay(model_name,attack_type, epsilon, num_steps, step_size, submit_time,gamma,perturbation_model,eps_min, eps_max, eps_points, target)
    
    meta = {
    "job_id": task.id,
    "model": model_name,
    "attack": attack_type,
    "submitted_at": str(submit_time),
    "hyperparameters_json": json.dumps({
        "epsilon": epsilon,
        "num_steps": num_steps,
        "step_size": step_size,
        "gamma": gamma,
        "perturbation_model": perturbation_model,
        "eps_min": eps_min,
        "eps_max": eps_max,
        "eps_points": eps_points,
        "target": target,

    }),
    }

    job_key = f"{JOB_KEY_PREFIX}{task.id}"

    # 1) store metadata for this job
    r.hset(job_key, mapping=meta)

    # 2) add to global index (newest first later)
    r.zadd(JOBS_INDEX_KEY, {task.id: submit_time})

    # optional: auto-expire metadata after 14 days
    r.expire(job_key, 14 * 24 * 3600)

    return {"job_id": task.id, "message": "Evaluation Task enqueued"}






# GET endpoint to check status
@app.get("/job_status/{job_id}")
def get_status(job_id: str):
    task_result = AsyncResult(job_id, app=create_task.app)
    return {
        "job_id": job_id,
        "status": task_result.status,
        "result": task_result.result
    }


def simplify_status(celery_status: str) -> str:
    # map Celery states to your UI states
    if celery_status in ("PENDING", "RECEIVED", "STARTED", "RETRY"):
        return "pending"
    if celery_status == "SUCCESS":
        return "success"
    if celery_status in ("FAILURE", "REVOKED"):
        return "failed"
    return celery_status.lower()


@app.get("/jobs")
def list_jobs(limit: int = Query(200, ge=1, le=2000)):
    rows = []

    # newest first
    job_ids = r.zrevrange(JOBS_INDEX_KEY, 0, limit - 1)

    for job_id in job_ids:
        job_key = f"{JOB_KEY_PREFIX}{job_id}"
        meta = r.hgetall(job_key)

        if not meta:
            # metadata missing (e.g. expired) -> clean index entry
            r.zrem(JOBS_INDEX_KEY, job_id)
            continue

        # decode hyperparameters
        try:
            hp = json.loads(meta.get("hyperparameters_json", "{}"))
        except json.JSONDecodeError:
            hp = {}

        # Celery status/result still comes from Celery backend (Redis DB 0)
        task_result = AsyncResult(job_id, app=create_task.app)
        celery_status = task_result.status

        rows.append({
            "job_id": job_id,
            "model": meta.get("model"),
            "attack": meta.get("attack"),
            "hyperparameters_of_attack": hp,
            "status": simplify_status(celery_status),
            "celery_status": celery_status,
        })

    return {"jobs": rows}





#Delete endpoint to remove a job
@app.delete("/delete_job/{job_id}")
def delete_job(job_id: str):
    r.delete(f"{JOB_KEY_PREFIX}{job_id}")
    r.zrem(JOBS_INDEX_KEY, job_id)

    # optional: delete celery result too
    task_result = AsyncResult(job_id, app=create_task.app)
    try:
        task_result.forget()
    except Exception:
        pass

    return {"job_id": job_id, "message": "Job deleted"}

