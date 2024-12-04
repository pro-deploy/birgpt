import os
import json
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# API URLs
CHAT_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I can help you with two things:\n"
        "1. Send any text message to get an AI response\n"
        "2. Use /img command followed by text to generate an image"
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "messages": [{"role": "user", "content": update.message.text}],
            "max_tokens": 500,
            "stream": False
        }
        
        response = requests.post(CHAT_API_URL, headers=headers, json=data)
        response_json = response.json()
        
        if response.status_code == 200:
            # Extract the response text from the model's output
            reply_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
            await update.message.reply_text(reply_text)
        else:
            await update.message.reply_text("Sorry, there was an error processing your request.")
            
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Extract the prompt from the command
        prompt = update.message.text.replace('/img', '').strip()
        if not prompt:
            await update.message.reply_text("Please provide a description after /img command")
            return

        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt
        }
        
        response = requests.post(IMAGE_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            # Send the image directly to telegram
            await update.message.reply_photo(response.content)
        else:
            await update.message.reply_text("Sorry, there was an error generating the image.")
            
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("img", generate_image))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()