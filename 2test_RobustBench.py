import torch
from robustbench.utils import load_model
import traceback
import urllib.request
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


def prototype_test_RobustBench():
    print("--- Loading Model from ROBUSTBENCH --- \n")


    cifar10_labels = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # 1. Scarica e carica il modello da RobustBench
    # Esempio: Un modello WideResNet robusto addestrato su CIFAR-10,
    # Nota: threat_model='Linf' indica che il modello è difeso contro attacchi L-infinito
    print(f"Scaricamento modello da RobustBench in corso...")
    try:
        model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf')
        # 2. Imposta il modello in modalità valutazione
        model.eval()
        print("\n Modello scaricato e caricato con successo!")

        # Stampa l'architettura
        # print(model)
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        traceback.print_exc()
        return


    print("\n--- Wrapping with secml-torch ---")
    try:
        secml_model = BasePytorchClassifier(model)
        print(f"\n Success: Model wrapped with secml-torch! \n ")
    except Exception as e:
        print(f"Error wrapping model: {e}")
        traceback.print_exc()
        return


    print("--- Running Prediction Test ---")
    try:
        print("\n Creating a dummy input image...")
        # 4. Test di predizione (CIFAR-10 ha immagini 32x32, non 224x224!)

        # Create a fake image that is pure BLUE
        # Channel 0 = Red, Channel 1 = Green, Channel 2 = Blue
        dummy_input = torch.zeros(1, 3, 32, 32)
        dummy_input[:, 1, :, :] = 1.0  # Set BLUE channel to Max (1.0)

        output = secml_model.predict(dummy_input)
        predicted_class_id = output.item()
        print(
            f"\n Success: The model produced '{cifar10_labels[predicted_class_id]}' as prediction with ID '{predicted_class_id}' with shape {output.shape}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    prototype_test_RobustBench()