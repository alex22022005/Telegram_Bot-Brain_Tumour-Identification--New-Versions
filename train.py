import os
import torch
from ultralytics import YOLO

def verify_gpu_setup():
    """
    Checks if PyTorch can detect a CUDA-enabled GPU and exits if it cannot.
    Returns the device string ('cuda:0' or 'cpu') for the model.
    """
    print("--- Verifying GPU Setup ---")
    try:
        if not torch.cuda.is_available():
            print("\nERROR: PyTorch cannot detect a CUDA-enabled GPU.")
            print(f"Current PyTorch version: {torch.__version__}")
            print("This version is CPU-only. Training cannot continue on the GPU.")
            print("\nPlease follow the instructions to uninstall this version and reinstall the correct GPU version of PyTorch.")
            return None # Signal failure

        device_name = torch.cuda.get_device_name(0)
        print(f"SUCCESS: GPU detected!")
        print(f"Device: {device_name}")
        print("---------------------------\n")
        return 0  # Use device index 0 for the GPU

    except Exception as e:
        print(f"An error occurred during GPU verification: {e}")
        return None


def train_model():
    """
    Trains the YOLOv8 model on the downloaded brain tumor dataset, requiring a GPU.
    """
    # Verify GPU is available before doing anything else.
    device = verify_gpu_setup()
    if device is None:
        return # Stop execution if no GPU is found

    # Define paths
    data_yaml_path = os.path.join('Brain-Tumor-Detection-1', 'data.yaml')
    weights_path = 'yolov8n.pt'

    # Check for required files
    if not all(os.path.exists(p) for p in [data_yaml_path, weights_path]):
        print("Error: Missing 'data.yaml' or 'yolov8n.pt'.")
        print("Please run 'download_dataset.py' and 'setup.py' first.")
        return

    try:
        print(f"--- Starting Training on GPU (cuda:{device}) ---")
        model = YOLO(weights_path)

        # Train the model
        model.train(
            data=data_yaml_path,
            epochs=100,
            imgsz=640,
            device=device,
            project='runs/training',
            name='brain_tumor_yolov8_gpu'
        )
        
        print("\nTraining completed successfully!")
        print("Model weights are saved in 'runs/training/brain_tumor_yolov8_gpu/weights'")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == '__main__':
    train_model()

