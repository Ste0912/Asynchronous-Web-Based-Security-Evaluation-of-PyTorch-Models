from fastapi import FastAPI
from celery.result import AsyncResult
from worker import create_dummy_task
from worker import another_dummy_task

app = FastAPI()

# POST endpoint to submit a URL
@app.post("/submit_eval")
def submit_eval(model_url: str):
    # Send task to Redis
    task = create_dummy_task.delay(model_url)
    return {"job_id": task.id, "message": "Task enqueued"}

# GET endpoint to check status
@app.get("/job_status/{job_id}")
def get_status(job_id: str):
    task_result = AsyncResult(job_id, app=create_dummy_task.app)
    return {
        "job_id": job_id,
        "status": task_result.status,
        "result": task_result.result
    }

#Delete endpoint to remove a job
@app.delete("/delete_job/{job_id}")
def delete_job(job_id: str):
    task_result = AsyncResult(job_id, app=create_dummy_task.app)
    if task_result.state != 'PENDING':
        task_result.forget()

#use another_dummy_task
@app.post("/submit_another_task")
def submit_another_task(data: str):
    task = another_dummy_task.delay(data)
    return {"job_id": task.id, "message": "Another task enqueued"}