import os
import requests
from tqdm import tqdm

def download_model_weights():
    """
    Downloads the pre-trained YOLOv8n model weights from the official repository.
    This helps to avoid network errors during the training process.
    """
    file_name = 'yolov8n.pt'
    url = f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{file_name}'

    if os.path.exists(file_name):
        print(f"'{file_name}' already exists. Skipping download.")
        return

    print(f"Downloading pre-trained model weights: {file_name}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during download.")
        else:
            print(f"Successfully downloaded '{file_name}'.")

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while trying to download the model.")
        print(f"Error details: {e}")
        print("\nPlease check your internet connection, firewall, or proxy settings.")
        print(f"Alternatively, you can manually download the file from:\n{url}\nand place it in the same directory as this script.")

if __name__ == '__main__':
    download_model_weights()
