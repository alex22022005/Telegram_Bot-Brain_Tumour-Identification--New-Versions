# Brain Tumor Detection & Information Bot

This project is a sophisticated Telegram bot designed to assist in the preliminary analysis of brain MRI scans. It leverages a custom-trained YOLOv8 object detection model to identify potential tumors and integrates with Google's Gemini API to provide informational responses to user queries.

### 🤖 [Try the Live Bot on Telegram](https://t.me/tumour_identification_bot) 🤖



## Features

-   **AI-Powered Tumor Detection**: Analyzes brain MRI scan images to detect three types of tumors: **Glioma**, **Meningioma**, and **Pituitary tumors**.
-   **Clear Classification**: Accurately identifies a **"No-Tumor"** case if no abnormalities are found.
-   **Severity Indication**: Provides a general severity level (High, Medium, Low) associated with detected tumor types.
-   **Visual Annotation**: Returns the original image with bounding boxes drawn around any potential findings for easy visualization.
-   **Conversational AI**: Users can ask follow-up questions about the detected tumor types, and the bot will provide general, educational information using Google's powerful Gemini model.
-   **Important Medical Disclaimer**: The bot consistently reminds users that it is an AI tool and not a substitute for professional medical advice.

## Technology Stack

-   **Backend**: Python
-   **Machine Learning**: YOLOv8 (Ultralytics) on PyTorch
-   **Telegram Bot Framework**: `python-telegram-bot`
-   **Generative AI**: Google Gemini API
-   **Dataset Management**: Roboflow

## Setup and Installation

Follow these steps to get the project running on your local machine.

#### 1. Prerequisites

-   Git
-   Python 3.10 or newer
-   An NVIDIA GPU with CUDA installed (highly recommended for training)

#### 2. Clone the Repository
## git clone
```bash
 https://github.com/alex22022005/Telegram_Bot-Brain_Tumour-Identification--New-Versions.git

```
#### 3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv

source .venv/bin/activate
```

