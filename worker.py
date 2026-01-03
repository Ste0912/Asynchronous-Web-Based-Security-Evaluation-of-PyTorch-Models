from celery import Celery
import torch
import traceback
import torchvision
import time
import tracemalloc      # For memory tracking
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from robustbench.utils import load_model
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier
from secmlt.adv.evasion.pgd import PGD

# Configure Celery to use the running Redis server
celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)


@celery_app.task(name="create_dummy_task")
def create_dummy_task(model_name, epsilon, num_steps, step_size):
    """
    This task performs the Month 2 logic:
    1. Loads the model specified by the API (model_name).
    2. Downloads REAL CIFAR-10 images.
    3. Runs a PGD Adversarial Attack.
    """

    # Use the variable passed from the API, do NOT use input()
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
        print(f"Worker: Downloading/Loading {model_to_load} from RobustBench...")
        # We assume dataset='cifar10' and threat_model='Linf'
        model = load_model(model_name=model_to_load, dataset='cifar10', threat_model='Linf')
        model.eval()
        print("\nWorker: Model loaded successfully.")
    except Exception as e:
        print(f"Worker Error during model loading: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }

    try:
        # --- PHASE 2: WRAP MODEL ---
        print("\nWorker: Wrapping model with secml-torch...")
        # BasePytorchClassifier handles the gradient tracking for the attack
        secml_model = BasePytorchClassifier(model)
        print("\nWorker: Model wrapped successfully.")
    except Exception as e:
        print(f"\nWorker Error during model wrapping: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }

    try:
        # --- PHASE 3: LOAD REAL DATA (CIFAR-10) ---
        print("Worker: Loading real CIFAR-10 test images...")

        # We use torchvision to download the official test set.
        # root='./data': Saves files to a 'data' folder in the project
        # train=False: We want the TEST set (for evaluation), not training data
        # transform=ToTensor(): Converts images to PyTorch Tensors [0, 1]
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=ToTensor()
        )

        # Select the first 4 images from the dataset
        batch_size = 4
        indices = range(batch_size)

        # Extract images and labels
        images_list = [test_dataset[i][0] for i in indices]
        labels_list = [test_dataset[i][1] for i in indices]

        # Stack them into a single batch Tensor: Shape (4, 3, 32, 32)
        images = torch.stack(images_list)
        true_labels = torch.tensor(labels_list)

        print(f"Worker: Batch created. True Labels: {[cifar10_labels[l] for l in true_labels]}")

    except Exception as e:
        print(f"Worker Error during data loading: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }

    try:
        # --- PHASE 4: CHECK CLEAN ACCURACY ---
        # Before attacking, see if the model gets them right on normal images
        clean_preds = secml_model.predict(images)

        # Calculate Clean Accuracy
        clean_correct = (clean_preds == true_labels).sum().item()
        clean_acc_percent = (clean_correct / batch_size) * 100
        print(f"Worker: Clean Accuracy: {clean_acc_percent}%")
    except Exception as e:
        print(f"Worker Error during clean accuracy evaluation: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }

    try:
        # --- PHASE 5: RUN PGD ATTACK ---
        print("Worker: Running PGD Attack (eps={epsilon}, steps={num_steps})...")

        # PGD L-infinity attack
        attack = PGD(
            perturbation_model="linf",
            epsilon=float(epsilon),
            num_steps=int(num_steps),
            step_size=float(step_size)
        )

        # --- Create a DataLoader for the Attack ---
        # The library expects an iterable that yields (image_batch, label_batch)
        dataset = TensorDataset(images, true_labels)
        attack_loader = DataLoader(dataset, batch_size=batch_size)

        # Run the attack
        # WE PASS TWO ARGUMENTS: (Model, DataLoader)
        # It returns a new DataLoader containing the adversarial images
        adv_loader = attack(secml_model, attack_loader)

        # Extract the adversarial images from the returned loader
        # Since we only have 1 batch, we just grab the first item
        adversarial_images, _ = next(iter(adv_loader))

        print("Worker: PGD Attack completed.")

        print("Worker: PGD Attack completed.")
    except Exception as e:
        print(f"Worker Error during PGD attack: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }
    try:
        # --- PHASE 6: EVALUATE ROBUSTNESS ---
        print("Worker: Evaluating attack impact...")

        # Get predictions on the attacked images
        adv_preds = secml_model.predict(adversarial_images)

        # Robust Accuracy: How many still match the TRUE label?
        robust_correct = (adv_preds == true_labels).sum().item()
        robust_accuracy = (robust_correct / batch_size) * 100

        # --- Performance Metrics Calculation ---
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_tracemalloc_memory()
        tracemalloc.stop()

        duration_sec = end_time - start_time
        peak_mem_mb = peak_mem / (1024 * 1024)  # Convert Bytes to MB

        print(f"Worker: Task finished in {duration_sec:.2f}s, Peak Memory: {peak_mem_mb:.2f}MB")

        print(f"Worker: Robust Accuracy = {robust_accuracy}%")

        # Format results for the dashboard
        # We convert indices (e.g., 3) to names (e.g., 'cat') for readability
        true_names = [cifar10_labels[i] for i in true_labels.tolist()]
        adv_names = [cifar10_labels[i] for i in adv_preds.tolist()]

        return {
            "status": "Completed",
            "model_name": model_name,
            "clean_accuracy": f"{clean_acc_percent:.1f}%",
            "robust_accuracy": f"{robust_accuracy:.1f}%",
            "true_labels": true_names,
            "adversarial_labels": adv_names,
            "attack_type": "PGD (L-inf)",
            "duration_sec": f"{duration_sec:.2f}",
            "memory_peak_mb": f"{peak_mem_mb:.2f}",
            "epsilon": epsilon,
            "verification": "Evaluation Completed"
        }

    except Exception as e:
        print(f"Worker Error: {e}")
        traceback.print_exc()
        return {
            "status": "Failed",
            "error": str(e)
        }





