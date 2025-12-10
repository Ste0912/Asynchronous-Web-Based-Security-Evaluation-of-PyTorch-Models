from celery import Celery
import time

# Configure Celery to use the running Redis server
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# The "Dummy Task" testing
@celery_app.task(name="create_dummy_task")
def create_dummy_task(model_url):
    print(f"Worker: Received task for {model_url}")
    time.sleep(10)  # Simulate a 10-second security analysis
    return {"status": "Completed", "model_url": model_url, "score": 0.85}

@celery_app.task(name="another_dummy_task")
def another_dummy_task(data):
    print(f"Worker: Processing data {data}")
    time.sleep(5)