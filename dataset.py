# Import the Roboflow library
from roboflow import Roboflow

# --- IMPORTANT ---
# It's best practice to use environment variables or a secrets management tool
# for your API key instead of hardcoding it directly in the script.
# For this example, we are using the key you provided.
API_KEY = "GUeccORHJBVBaiFUIxSy"

def download_brain_tumor_dataset():
    """
    Downloads the brain tumor detection dataset from Roboflow.
    """
    try:
        # Initialize the Roboflow object with your API key
        rf = Roboflow(api_key=API_KEY)

        # Access your workspace and project
        project = rf.workspace("workspace-kjrwm").project("brain-tumor-detection-6prsv")

        # Select a specific version of your dataset
        version = project.version(1)

        # Download the dataset in YOLOv8 format
        # This will create a folder named 'Brain-Tumor-Detection-1' in your project directory
        dataset = version.download("yolov8")
        print("Dataset downloaded successfully!")
        print(f"Dataset location: {dataset.location}")
        return dataset.location
    except Exception as e:
        print(f"An error occurred during dataset download: {e}")
        return None

if __name__ == "__main__":
    download_brain_tumor_dataset()
