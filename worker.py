from celery import Celery
import torch
import traceback
import torchvision
import redis
import json
import time
import tracemalloc      # For memory tracking
import io
import os
import psutil
import base64
import math
import numpy as np
import h5py
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from robustbench.utils import load_model
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.adv.evasion.pgd import PGD
from secmlt.adv.evasion.fmn import FMN
from torch.utils.data import Dataset
#from dataset_loader import H5Dataset
from robustbench_utils import get_robustbench_point


# Configure Celery to use the running Redis server
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# --- Redis DB1 for durable storage 
APP_REDIS_URL = os.getenv("APP_REDIS_URL", "redis://localhost:6379/1")
app_r = redis.Redis.from_url(APP_REDIS_URL, decode_responses=True)

JOB_KEY_PREFIX = "secml:job:"

def persist_final_result(job_id: str, celery_status: str, result_payload: dict):
    job_key = f"{JOB_KEY_PREFIX}{job_id}"
    app_r.hset(job_key, mapping={
        "stored_status": celery_status,
        "finished_at": str(time.time()),
        # default=str avoids crash if something is not JSON-serializable
        "result_json": json.dumps(result_payload),
    })



def get_device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"



def fail_and_persist(job_id: str, error: Exception):
    payload = {"status": "Failed", "error": str(error)}
    persist_final_result(job_id, "FAILURE", payload)
    return payload



def batch_lp_norm(delta: torch.Tensor, p_model: str) -> torch.Tensor:
    """
    delta: (N, C, H, W)
    returns: (N,) per-sample norms
    """
    flat = delta.view(delta.shape[0], -1).abs()

    if p_model == "linf":
        return flat.max(dim=1).values
    elif p_model == "l2":
        return torch.sqrt((flat ** 2).sum(dim=1))
    elif p_model == "l1":
        return flat.sum(dim=1)
    else:
        raise ValueError(f"Unsupported perturbation_model: {p_model}")



@celery_app.task(bind=True, name="create_task")
def create_task(self, model_name,attack_type, epsilon, num_steps, step_size, submit_time,gamma=0.05,perturbation_model="linf",eps_min=0.0, eps_max=0.1, eps_points=21,target=None):
    job_id = self.request.id
    process = psutil.Process(os.getpid())
    def rss_mb():
        return process.memory_info().rss / (1024 * 1024)
    rss_start_mb = rss_mb()
   
    rss_peak_mb = rss_start_mb

    
    # Capture the exact time the worker actually starts processing
    start_time_queue = time.time()

    #Calculate Queue Wait Time that is the Queue Performance
    # The difference between now (start) and when it was sent (submit)
    queue_wait_seconds = start_time_queue - submit_time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nWorker: Using device = {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


    # Use monotonic for durations (not affected by clock adjustments)
    t0_total = time.monotonic()
    
    # Use the variable passed from the API
    model_to_load = model_name

    # Start tracking performance
    start_time = time.time()
    tracemalloc.start()

    print(f"\nWorker: Starting evaluation for model: {model_to_load}")

    
    # Define labels for CIFAR-10 (standard for RobustBench)
    cifar10_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    

    try:
        # --- PHASE 1: LOAD MODEL ---
        t0_model = time.monotonic()
        print(f"\n Worker: Downloading/Loading {model_to_load} from RobustBench...")
       
        # We assume dataset='cifar10' and threat_model='Linf'
        model = load_model(model_name=model_to_load, dataset='cifar10', threat_model='Linf')
        model.eval()
        model = model.to(device)

        t1_model = time.monotonic()
        print("\nWorker: Model loaded successfully.")
        rss_peak_mb = max(rss_peak_mb, rss_mb())

    except Exception as e:
        print(f"Worker Error during model loading: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)


    try:
        # --- PHASE 2: WRAP MODEL ---
       
        print("\nWorker: Wrapping model with secml-torch...")
        # BasePytorchClassifier handles the gradient tracking for the attack
        secml_model = BasePytorchClassifier(model)
        print("\nWorker: Model wrapped successfully.")
    except Exception as e:
        print(f"\nWorker Error during model wrapping: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)


    try:
        # --- PHASE 3: LOAD REAL DATA (CIFAR-10) ---
        t0_data = time.monotonic()
        print("Worker: Loading real CIFAR-10 test images...")

        
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=ToTensor()
        )

        # Select the first 10 images from the dataset
        batch_size = 20
        indices = range(batch_size)

        # Extract images and labels
        images_list = [test_dataset[i][0] for i in indices]
        labels_list = [test_dataset[i][1] for i in indices]

        # Stack them into a single batch Tensor: Shape (10, 3, 32, 32)
        images = torch.stack(images_list)
        true_labels = torch.tensor(labels_list)
        images = images.to(device)  
        true_labels = true_labels.to(device)


        t1_data = time.monotonic()
        print(f"\nWorker: Batch created. True Labels: {[cifar10_labels[l] for l in true_labels]}")
        rss_peak_mb = max(rss_peak_mb, rss_mb())

    except Exception as e:
        print(f"\n Worker Error during data loading: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)


    try:
        # --- PHASE 4: CHECK CLEAN ACCURACY ---
        # Before attacking, see if the model gets them right on normal images
        clean_preds = secml_model.predict(images)

        # Calculate Clean Accuracy
        clean_correct = (clean_preds == true_labels).sum().item()
        clean_acc_percent = (clean_correct / batch_size) * 100
        print(f"\n Worker: Clean Accuracy: {clean_acc_percent}%")
        
        rss_peak_mb = max(rss_peak_mb, rss_mb())
        rss_peak_upto_clean_mb = rss_peak_mb

    except Exception as e:
        print(f"Worker Error during clean accuracy evaluation: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)


    try:
        # --- PHASE 5: RUN PGD ATTACK ---
        t0_attack = time.monotonic()
        print(f"\n Worker: Running {attack_type} Attack ")

        attack = None
        rb_point = None

        # SELECT ATTACK
        if attack_type.startswith("pgd"):
            # Extract 'linf', 'l2', or 'l1' from string 'pgd-linf'
            p_model = attack_type.split("-")[1]
            if(p_model == "linf"):
                # PGD L-infinity attack
                attack = PGD(
                    perturbation_model=p_model,
                    epsilon=float(epsilon),
                    num_steps=int(num_steps),
                    step_size=float(step_size)
                )
            elif(p_model == "l2"):
                # PGD L2 Attack
                attack = PGD(
                    perturbation_model="l2",
                    epsilon=float(epsilon),
                    num_steps=int(num_steps),
                        step_size=float(step_size)
                    )
                
            elif(p_model == "l1"):
                # PGD L1 Attack
                attack = PGD(
                perturbation_model="l1",
                epsilon=float(epsilon),
                num_steps=int(num_steps),
                step_size=float(step_size)
                )
                pass

        elif attack_type.startswith("fmn"):
            # Extract 'linf', 'l2', or 'l1' from string 'fmn-linf'
            p_model = attack_type.split("-")[1]
            rb_point = get_robustbench_point(model_to_load, p_model)
            y_target = int(target) if target is not None else None

            if(p_model == "linf"):
                # FMN L-infinity Attack
                attack = FMN(
                    perturbation_model="linf",
                    num_steps=int(num_steps),
                    step_size=float(step_size),
                    gamma=float(gamma),
                    y_target=y_target,
                )
            elif(p_model == "l2"):
                # FMN L2 Attack
                attack = FMN(
                    perturbation_model="l2",
                    num_steps=int(num_steps),
                    step_size=float(step_size),
                    gamma=float(gamma),
                    y_target=y_target,
                )
            elif(p_model == "l1"):
                # FMN L1 Attack
                attack = FMN(
                    perturbation_model="l1",
                    num_steps=int(num_steps),
                    step_size=float(step_size),
                    gamma=float(gamma),
                    y_target=y_target,
                )

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")




        # --- Create a DataLoader for the Attack ---
        # The library expects an iterable that yields (image_batch, label_batch)
        dataset = TensorDataset(images, true_labels)
        attack_loader = DataLoader(dataset, batch_size=batch_size)


        # Run the attack
        # WE PASS TWO ARGUMENTS: (Model, DataLoader)
        # It returns a new DataLoader containing the adversarial images
        if attack_type.startswith("fmn") :
            adv_loader = attack(secml_model, attack_loader)
               

        # Extract the adversarial images from the returned loader
        # Since we only have 1 batch, we just grab the first item
        adversarial_images, _ = next(iter(adv_loader))
        t1_attack = time.monotonic()
        print(f"\n Worker: {attack_type} Attack completed.")
        rss_peak_mb = max(rss_peak_mb, rss_mb())

    except Exception as e:
        print(f"Worker Error during PGD attack: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)

    try:
        # --- PHASE 6: EVALUATE ROBUSTNESS ---
        print("\n Worker: Evaluating attack impact...")

        
        # Get predictions on the attacked images
        adv_preds = secml_model.predict(adversarial_images)

        curve_payload = None  # default: no curve

        # --- If FMN: build robust accuracy curve vs epsilon ---
        if attack_type.startswith("fmn"):
            # which norm? "fmn-linf" -> "linf"
            p_model = attack_type.split("-")[1]

            # Clean correctness mask (robustness is usually counted only for clean-correct samples)
            clean_correct_mask = (clean_preds == true_labels)          # (N,)

            # Attack success means adv prediction != true label
            if target is None:
                attack_success_mask = (adv_preds != true_labels)   # untargeted
            else:
                attack_success_mask = (adv_preds == int(target))   # targeted success


            # Compute per-sample perturbation size ||adv - orig||_p
            delta = (adversarial_images - images).detach()
            dists = batch_lp_norm(delta, p_model)                      # (N,)

            # If attack failed, treat required distance as +inf (robust for all eps)
            inf = torch.tensor(float("inf"), device=dists.device)
            effective_dists = torch.where(attack_success_mask, dists, inf)


            # JSON cannot represent inf; replace with a large sentinel
            INF_SENTINEL = 1e9
            effective_dists_json = effective_dists.detach().cpu().numpy()
            effective_dists_json = np.where(np.isinf(effective_dists_json), INF_SENTINEL, effective_dists_json).tolist()

            clean_correct_json = clean_correct_mask.detach().cpu().to(torch.int).numpy().tolist()

            curve_detail = {
                "sample_indices": list(range(batch_size)),   
                "clean_correct_mask": clean_correct_json,    # 0/1
                "effective_dists": effective_dists_json,     # float list
                "inf_sentinel": INF_SENTINEL
            }







            # Epsilon grid
            eps_grid = np.linspace(float(eps_min), float(eps_max), int(eps_points)).tolist()

            # Robust accuracy curve (percentage over ALL samples, matching your current style)
            rob_curve = []
            for eps in eps_grid:
                eps_t = torch.tensor(eps, device=effective_dists.device)
                robust_mask = clean_correct_mask & (effective_dists > eps_t)
                robust_acc = (robust_mask.sum().item() / batch_size) * 100.0
                rob_curve.append(round(robust_acc, 3))

            # Define "robust_accuracy" as the last point (at eps_max)
            robust_accuracy = rob_curve[-1]

            curve_payload = {
                "epsilons": eps_grid,
                "robust_accuracy": rob_curve,
                "perturbation_model": p_model,
                "eps_min": float(eps_min),
                "eps_max": float(eps_max),
                "eps_points": int(eps_points),
                "detail": curve_detail
            }

        else:
            # --- Non-FMN ( original single robust accuracy) ---
            robust_correct = (adv_preds == true_labels).sum().item()
            robust_accuracy = (robust_correct / batch_size) * 100.0

                
       



        # --- Performance Metrics Calculation ---
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        

        duration_sec = end_time - start_time
        peak_mem_mb = peak_mem / (1024 * 1024)  # Convert Bytes to MB

        print(f"\nWorker: Task finished in {duration_sec:.2f}s, Peak Memory: {peak_mem_mb:.2f}MB")

        print(f"\n Worker: Robust Accuracy = {robust_accuracy}%")

        # Format results for the dashboard
        # We convert indices (e.g., 3) to names (e.g., 'cat') for readability
        true_names = [cifar10_labels[i] for i in true_labels.tolist()]
        adv_names = [cifar10_labels[i] for i in adv_preds.tolist()]

    except Exception as e:
        print(f"Worker Error: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)


    try:
        # --- PHASE 7: PREPARE VISUALIZATION (Sample 0) ---
        t0_viz = time.monotonic()
        print("\n Worker: Generating visualization images...")

        # We only take the first image in the batch (index 0)
        orig_tensor = images[0]
        adv_tensor = adversarial_images[0]

        # Calculate perturbation (noise)
        noise_tensor = adv_tensor - orig_tensor

        # Convert to Base64
        img_orig_b64 = tensor_to_base64(orig_tensor)
        img_adv_b64 = tensor_to_base64(adv_tensor)
        img_noise_b64 = tensor_to_base64(noise_tensor, amplify=True)
        
        t1_viz = time.monotonic()
        rss_peak_mb = max(rss_peak_mb, rss_mb())

        # Stop timers
        t1_total = time.monotonic()

        # tracemalloc
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration_sec = t1_total - t0_total

        timing = {
            "total_sec": round(duration_sec, 3),
            "model_load_sec": round(t1_model - t0_model, 3),
            "data_load_sec": round(t1_data - t0_data, 3),
            "attack_sec": round(t1_attack - t0_attack, 3),
            "viz_sec": round(t1_viz - t0_viz, 3),
    }
        rss_end_mb = process.memory_info().rss / (1024 * 1024)

        final_payload={
            "status": "Completed",
            "model_name": model_name,
            "device_name": get_device_name(device),
            "clean_accuracy": f"{clean_acc_percent:.1f}%",
            "robust_accuracy": f"{robust_accuracy:.1f}%",
            "curve": curve_payload,
            "true_labels": true_names,
            "adversarial_labels": adv_names,
            "queue_wait_sec": round(queue_wait_seconds, 4),
            "timing": timing,
            "py_peak_alloc_mb": round(peak_mem / (1024 * 1024), 2),
            "attack_type": f"{attack_type}",
            "duration_sec": f"{duration_sec:.2f}",
            "memory_peak_mb": f"{peak_mem_mb:.2f}",
            "rss_start_mb": round(rss_start_mb, 2),
            "rss_end_mb": round(rss_end_mb, 2),
            "rss_peak_mb": round(rss_peak_mb, 2),
            "rss_peak_upto_clean_mb": round(rss_peak_upto_clean_mb, 2),

            "epsilon": epsilon,
            "num_steps": num_steps,
            "gamma": gamma,
            "step_size": step_size,
            "robustbench_point": rb_point,
            "fmn_target": (int(target) if target is not None else None),
            "attack_label": f"{attack_type} | target={target}",

             # Add images to the response
            "images": {
                "original": img_orig_b64,
                "adversarial": img_adv_b64,
                "noise": img_noise_b64
            }
        }

        persist_final_result(job_id, "SUCCESS", final_payload)
        return final_payload



    except Exception as e:
        print(f"Worker Error: {e}")
        traceback.print_exc()
        return fail_and_persist(job_id, e)





def tensor_to_base64(tensor, amplify=False):
    """Converts a PyTorch tensor to a Base64 PNG string."""
    # Convert (C, H, W) -> (H, W, C) and move to CPU/Numpy
    img_np = tensor.cpu().detach().numpy().transpose(1, 2, 0)

    if amplify:
        # Normalize to [0, 1] so the noise is visible
        if img_np.max() > img_np.min():
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    else:
        # Clip standard images to valid range [0, 1]
        img_np = np.clip(img_np, 0, 1)

    # Convert to 8-bit [0, 255]
    img_np = (img_np * 255).astype(np.uint8)

    # Save to BytesIO buffer
    img_pil = Image.fromarray(img_np)
    buff = io.BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")



