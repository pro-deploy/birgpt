import os
import json
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio
import aiohttp

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# API URLs
CHAT_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Initialize vector store dictionary
user_vector_stores = {}

def process_document(file_path):
    # Determine the loader based on file extension
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    
    return vector_store

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I can help you with:\n"
        "1. Send any text message to get an AI response\n"
        "2. Use /img command followed by text to generate an image\n"
        "3. Send PDF, TXT, or DOCX files to analyze them and ask questions"
    )

async def send_typing_action(context, chat_id):
    while True:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)  # Telegram's typing status lasts 5 seconds
        except asyncio.CancelledError:
            break

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.effective_chat.id

    # Start typing action task
    typing_task = asyncio.create_task(send_typing_action(context, chat_id))
    
    try:
        if user_id in user_vector_stores:
            try:
                vector_store = user_vector_stores[user_id]
                retriever = vector_store.as_retriever()
                
                docs = retriever.get_relevant_documents(update.message.text)
                context_text = "\n".join([doc.page_content for doc in docs])
                
                enhanced_prompt = f"Context: {context_text}\n\nQuestion: {update.message.text}"
                
                headers = {
                    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "max_tokens": 500,
                    "stream": False
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(CHAT_API_URL, headers=headers, json=data) as response:
                        response_json = await response.json()
                        
                        if response.status == 200:
                            reply_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
                            await update.message.reply_text(reply_text)
                        else:
                            await update.message.reply_text("Sorry, there was an error processing your request.")
                
            except Exception as e:
                await update.message.reply_text(f"An error occurred: {str(e)}")
        else:
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
            
            async with aiohttp.ClientSession() as session:
                async with session.post(CHAT_API_URL, headers=headers, json=data) as response:
                    response_json = await response.json()
                    
                    if response.status == 200:
                        reply_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
                        await update.message.reply_text(reply_text)
                    else:
                        await update.message.reply_text("Sorry, there was an error processing your request.")
    
    finally:
        # Stop typing action
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Start typing action task
    typing_task = asyncio.create_task(send_typing_action(context, chat_id))
    
    try:
        file = await context.bot.get_file(update.message.document.file_id)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(update.message.document.file_name)[1]) as tmp_file:
            await file.download_to_drive(tmp_file.name)
            
            vector_store = process_document(tmp_file.name)
            
            user_id = update.message.from_user.id
            user_vector_stores[user_id] = vector_store
            
            os.unlink(tmp_file.name)
            
        await update.message.reply_text(
            "Document processed successfully! You can now ask questions about its content."
        )
        
    except Exception as e:
        await update.message.reply_text(f"Error processing document: {str(e)}")
    
    finally:
        # Stop typing action
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    
    # Start typing action task
    typing_task = asyncio.create_task(send_typing_action(context, chat_id))
    
    try:
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
        
        async with aiohttp.ClientSession() as session:
            async with session.post(IMAGE_API_URL, headers=headers, json=data) as response:
                if response.status == 200:
                    image_data = await response.read()
                    await update.message.reply_photo(image_data)
                else:
                    await update.message.reply_text("Sorry, there was an error generating the image.")
    
    except Exception as e:
        await update.message.reply_text(f"An error occurred: {str(e)}")
    
    finally:
        # Stop typing action
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("img", generate_image))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()