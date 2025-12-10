import torch
import traceback
import torchvision.models as models
from secmlt.models.pytorch.base_pytorch_nn import BasePytorchClassifier


def prototype_test():
    print("--- Phase 1: Loading Model from Torch Hub ---")
    # List of some available models in torchvision
    #To make the strings more readable after printing we can skip one line
    print("\n")

    print("The list of some available models is the following:")
    print("\n")
    available_models = [
        'resnet18', 'alexnet', 'vgg16', 'squeezenet1_0', 'densenet121',
        'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2',
        'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet1_0'
    ]
    for index,model_name in enumerate(available_models, start=1):
        print(f"-{index} {model_name}")
    print("\n")
    print("Type the name of the model to load:")
    model_to_load = input().strip()
    print("\n")
    try:
        # Load model with pre-trained weights
        model = torch.hub.load('pytorch/vision', model_to_load, weights='DEFAULT')
        model.eval()
        print("Success: model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n--- Phase 2: Wrapping with secml-torch ---")
    try:
        secml_model = BasePytorchClassifier(model)
        print(f"Success: Model {model_to_load} wrapped in BasePytorchClassifier ().")
    except Exception as e:
        print(f"Error wrapping model: {e}")
        return

    print("\n--- Phase 3: Running Prediction Test ---")
    try:
        # Create a dummy image (Batch size 1, 3 Channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)

        # Run prediction through the security wrapper
        # The wrapper expects the input to be on the correct device (CPU by default here)
        output = secml_model.predict(dummy_input)

        print(f"Success: The model produced a prediction with shape {output.shape}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        # Detailed error printing to help debug if mismatch occurs
        traceback.print_exc()


if __name__ == "__main__":
    prototype_test()