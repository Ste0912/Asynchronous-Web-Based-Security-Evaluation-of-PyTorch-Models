import time
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from celery.result import AsyncResult
from worker import create_task


app = FastAPI()

# --- Serve the "static" folder at the "/static" URL ---
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# In-memory registry: job_id -> metadata
JOB_STORE: Dict[str, Dict[str, Any]] = {}

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
        eps_points: int = 21
        
                ):
   
    #Capture the timestamp of submission (Enqueue Time)
    submit_time = time.time()

    task = create_task.delay(model_name,attack_type, epsilon, num_steps, step_size, submit_time,gamma,perturbation_model,eps_min, eps_max, eps_points)
    
     # record job metadata so we can show it later in the table
    JOB_STORE[task.id] = {
        "job_id": task.id,
        "model": model_name,
        "attack": attack_type,
        "hyperparameters_of_attack": {
            "epsilon": epsilon,
            "num_steps": num_steps,
            "step_size": step_size,
            "gamma": gamma,
            "perturbation_model": perturbation_model,
            "eps_min": eps_min,
            "eps_max": eps_max,
            "eps_points": eps_points,
        },
        "submitted_at": submit_time,
    }   
    
    
    
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
def list_jobs():
    rows = []

    # return newest first (optional)
    items = sorted(JOB_STORE.values(), key=lambda x: x["submitted_at"], reverse=True)

    for meta in items:
        job_id = meta["job_id"]
        task_result = AsyncResult(job_id, app=create_task.app)
        celery_status = task_result.status

        rows.append({
            "job_id": job_id,
            "model": meta["model"],
            "attack": meta["attack"],
            "hyperparameters_of_attack": meta["hyperparameters_of_attack"],
            "status": simplify_status(celery_status),
            "celery_status": celery_status,
        })

    return {"jobs": rows}




#Delete endpoint to remove a job
#@app.delete("/delete_job/{job_id}")
#def delete_job(job_id: str):
#    task_result = AsyncResult(job_id, app=create_dummy_task.app)
#    if task_result.state != 'PENDING':
#        task_result.forget()
#    return {"job_id": job_id, "message": "Job deleted"}

