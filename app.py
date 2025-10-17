import os
import io
import logging
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.constants import ParseMode
# UPDATED: Corrected imports for the newer library version
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters
from ultralytics import YOLO

# --- Setup ---
# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_PATH = 'runs/training/brain_tumor_yolov8_gpu/weights/best.pt' 

# UPDATED: Mapped your new model's class indices to readable names.
# Based on your dataset: 0:Glioma, 1:Meningioma, 2:No-Tumor, 3:Pituitary
# Please verify these are the correct names and order for your trained model.
CLASS_NAMES = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor',
    2: 'No-Tumor',
    3: 'Pituitary Tumor'
}

# UPDATED: Map class indices to severity levels for the new classes.
SEVERITY_MAPPING = {
    0: 'High Severity',
    1: 'Low Severity',
    3: 'Medium Severity'
    # Class 2 (No-Tumor) is handled separately and does not have a severity.
}
# --- End Configuration ---

# --- Load Models ---
try:
    # Load the custom YOLOv8 model for tumor detection
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at '{MODEL_PATH}'. Please ensure the path is correct.")
    yolo_model = YOLO(MODEL_PATH)
    logger.info("Brain Tumor Detection model loaded successfully.")

    # Configure the Gemini API for the chatbot
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini model configured successfully.")

except Exception as e:
    logger.critical(f"A critical error occurred during model initialization: {e}")
    exit()

# --- Telegram Bot Handlers ---

async def start(update: Update, context: CallbackContext) -> None:
    """Sends a welcome message when the /start command is issued."""
    user_name = update.effective_user.first_name
    welcome_message = (f"👋 Hello, {user_name}!\n\n"
                       "I am the Brain Tumor Detection Bot. Please send me a brain scan image, and I will analyze it for potential tumor types.")
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: CallbackContext) -> None:
    """Sends instructions when the /help command is issued."""
    help_text = ("**How to use me:**\n"
                 "1. **Analysis:** Send me a brain scan image (like an MRI).\n"
                 "2. **Follow-up:** After I provide the analysis, you can ask me questions about the detected tumor types for informational purposes.")
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def handle_image(update: Update, context: CallbackContext) -> None:
    """Processes a brain scan image, runs detection, and sends back the annotated result."""
    chat_id = update.message.chat_id
    try:
        await context.bot.send_message(chat_id, "Scan received. Analyzing... 🩺")

        # Download and open the image
        photo_file = await update.message.photo[-1].get_file()
        image_bytes_io = io.BytesIO()
        await photo_file.download_to_memory(image_bytes_io)
        image_bytes_io.seek(0)
        pil_image = Image.open(image_bytes_io)

        # Perform YOLOv8 inference
        results = yolo_model(pil_image)
        result = results[0] 
        
        # Render the results on the image
        annotated_image_np = result.plot()
        annotated_image_pil = Image.fromarray(annotated_image_np[..., ::-1])

        # Save the annotated image to a byte buffer
        bio = io.BytesIO()
        bio.name = 'annotated_scan.jpeg'
        annotated_image_pil.save(bio, 'JPEG')
        bio.seek(0)
        
        # UPDATED LOGIC: Handle 'No-Tumor' class separately
        detected_tumors = []
        detected_tumor_names_for_chatbot = []
        no_tumor_case_detected = False

        boxes = result.boxes
        if boxes and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES.get(class_id, "Unknown Type")

                if class_name == 'No-Tumor':
                    no_tumor_case_detected = True
                else:
                    severity = SEVERITY_MAPPING.get(class_id, "Unknown Severity")
                    detected_tumors.append(f"{class_name} ({severity})")
                    detected_tumor_names_for_chatbot.append(class_name)

        # Remove duplicates
        detected_tumors = list(dict.fromkeys(detected_tumors))
        detected_tumor_names_for_chatbot = list(dict.fromkeys(detected_tumor_names_for_chatbot))

        # Store context for the chatbot
        context.user_data['last_detected_tumors'] = detected_tumor_names_for_chatbot
        
        # Construct the caption based on findings
        if detected_tumors:
            findings_text = "\n".join([f"• {finding}" for finding in detected_tumors])
            caption_text = f"**Analysis Complete.**\n\nPotential findings:\n{findings_text}\n\nYou can now ask me questions for more information."
        elif no_tumor_case_detected:
            caption_text = "Analysis complete. The scan indicates no tumor was found."
            context.user_data['last_detected_tumors'] = None
        else:
            caption_text = "Analysis complete. I did not detect any of the conditions I'm trained to recognize in this scan."
            context.user_data['last_detected_tumors'] = None

        await update.message.reply_photo(photo=bio, caption=caption_text, parse_mode=ParseMode.MARKDOWN)
        log_findings = ', '.join(detected_tumors) if detected_tumors else ('No-Tumor' if no_tumor_case_detected else 'None')
        logger.info(f"Processed scan for chat {chat_id}. Found: {log_findings}")

    except Exception as e:
        logger.error(f"Error processing image for chat {chat_id}: {e}", exc_info=True)
        await update.message.reply_text("Sorry, I encountered an error during the analysis. Please try sending the scan again.")


async def handle_text(update: Update, context: CallbackContext) -> None:
    """Handles text messages and uses Gemini for chatbot functionality."""
    user_question = update.message.text
    chat_id = update.message.chat_id
    
    thinking_message = await context.bot.send_message(chat_id=chat_id, text="Consulting knowledge base... 🧠")

    last_tumors = context.user_data.get('last_detected_tumors')
    
    prompt_context = "A user is asking a general question about brain tumors."
    if last_tumors:
        tumor_list = ', '.join(last_tumors)
        prompt_context = (f"A user's brain scan analysis has indicated a potential '{tumor_list}'. "
                          f"The user is now asking a follow-up question.")

    prompt = (f"You are a helpful medical information AI. {prompt_context} "
              f"Answer their question in a clear, simple, and reassuring tone. Provide general information only.\n\n"
              f"User's Question: '{user_question}'")
    
    disclaimer = "\n\n**Important Disclaimer:** I am an AI assistant, not a medical professional. This information is for educational purposes only. Please consult a qualified doctor for diagnosis and medical advice."

    try:
        response = gemini_model.generate_content(prompt)
        bot_answer = response.text + disclaimer
        
        await context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text=bot_answer, parse_mode=ParseMode.MARKDOWN)
        logger.info(f"Answered text query for chat {chat_id}.")

    except Exception as e:
        logger.error(f"Error calling Gemini API for chat {chat_id}: {e}", exc_info=True)
        error_message = "I'm having trouble accessing my knowledge base right now. Please try again later." + disclaimer
        await context.bot.edit_message_text(chat_id=chat_id, message_id=thinking_message.message_id, text=error_message, parse_mode=ParseMode.MARKDOWN)

def main() -> None:
    """Starts the Telegram bot."""
    if not TELEGRAM_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN not found in environment variables. Bot cannot start.")
        return

    # UPDATED: Use Application.builder() for initialization
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers to the application
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    # UPDATED: Run the bot using run_polling()
    logger.info("Brain Tumor Detection Bot is starting...")
    application.run_polling()


if __name__ == '__main__':
    main()

