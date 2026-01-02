from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from celery.result import AsyncResult
from worker import create_dummy_task


app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

#Redirect root URL (/) to the dashboard
@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# POST endpoint to submit a Model Name
@app.post("/submit_eval")
def submit_eval(model_name: str):
    # Send task to Redis,pass the input string directly to the worker
    task = create_dummy_task.delay(model_name)
    return {"job_id": task.id, "message": "Evaluation Task enqueued"}

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
    return {"job_id": job_id, "message": "Job deleted"}

