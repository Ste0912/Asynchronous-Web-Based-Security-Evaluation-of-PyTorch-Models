# Asynchronous Web-Based Security Evaluation of Machine Learning Models

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Celery](https://img.shields.io/badge/Celery-37814A?style=flat&logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat&logo=redis&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

A robust, asynchronous web platform designed to evaluate the security of PyTorch models against adversarial machine learning attacks. Built with a decoupled architecture, this tool offloads heavy adversarial optimization tasks to background workers, ensuring a responsive and non-blocking user interface.

## Key Features

* **Granular Attack Configuration:** Fine-tune attack hyperparameters directly from the UI, including epsilon limits, step sizes, and iterations for PGD, as well as gamma, targeting, and epsilon sweeps for FMN minimum-norm attacks.
* **Singular Security Evaluation Curves:** Automatically generate and visualize the robustness curve (accuracy vs. perturbation budget) for any individual attack run.
* **Dynamic Ensemble Construction:** Navigate the job registry to select specific attacks and combine them into a custom ensemble threat model.
* **Comparative Ensemble Visualizations:** Plot and compare the worst-case Ensemble Security Evaluation Curve directly against singular attack curves to identify true model vulnerabilities.
* **Asynchronous Processing:** Utilizes Celery and Redis to handle intensive Machine Learning workloads in the background without blocking the UI.
* **Qualitative Image Inspection:** Provides side-by-side visual comparisons of original inputs, adversarial examples, and amplified perturbation noise.
* **Comprehensive Reporting:** Automatically generates and exports ensemble evaluation metrics to CSV, PDF, Markdown, and LaTeX formats for academic and professional reporting.

## Architecture & Tech Stack

* **Backend API:** FastAPI
* **Task Queue & Broker:** Celery, Redis
* **Machine Learning:** PyTorch, Torchvision, SecML-Torch, RobustBench
* **Frontend:** Vanilla HTML/JS, Chart.js, jsPDF

## Installation & Setup

**Prerequisites:**
* Python 3.x
* Redis server (running locally on default port 6379)

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd /mnt/c/projects/Asynchronous_WebBased_Security_Evaluation_of_ML_Models

2. **Create and activate a virtual environment:**
  ```bash
python3 -m venv venv_wsl
source venv_wsl/bin/activate
 ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

##  Running the Application

Because this is an asynchronous application, you need to run the background task worker and the web server simultaneously in two separate terminal windows.

### Terminal 1: Start the Celery Worker
This process handles the downloading of models, data loading, and adversarial attack generation. Keep this window open.

```bash
python3 -m celery -A worker.celery_app worker --loglevel=info
```
*You should see the Celery banner and a `[tasks]` section indicating it is ready.*

### Terminal 2: Start the FastAPI Server
Open a new terminal window, navigate to the project, and activate the environment again.

```bash
cd /mnt/c/path/to/your/project
source venv_wsl/bin/activate
uvicorn main:app --reload
```
*You should see Uvicorn running on `http://127.0.0.1:8000`.*

## Usage & Analytical Workflow

1. **Configure & Launch:** Open `http://127.0.0.1:8000/static/index.html`. Select a model, choose an attack type, configure your specific hyperparameters, and submit the evaluation.
2. **Analyze Singular Attacks:** Once the asynchronous job completes, view the robust accuracy metrics, visual image inspection, and the singular Security Evaluation Curve.
3. **Build an Ensemble:** Navigate to the Job Registry (`/static/jobs.html`) to view the history of all successful evaluations. Use the checkboxes to select multiple FMN attacks with matching norms and sweep parameters.
4. **Compare Curves:** Click "View selected curve(s)" to generate an aggregate view. The dashboard will overlay the singular attack curves alongside the newly calculated, unified Ensemble Security Evaluation Curve.

## 📁 Project Structure

* `main.py`: FastAPI application routing, job submission, and status retrieval.
* `worker.py`: Celery worker configuration, model loading, and SecML-Torch attack execution.
* `static/index.html`: Main user dashboard for configuring attacks, tracking jobs, and visualizing comparative curves.
* `static/jobs.html`: Registry interface for viewing past attacks and selecting jobs for ensemble evaluation.
* `robustbench_utils.py`: Utilities for retrieving baseline robust accuracy metrics.
```

