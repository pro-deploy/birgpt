import os
import json
import requests
import tempfile
import logging
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

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# API URLs
CHAT_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

# Инициализация словаря векторных хранилищ
user_vector_stores = {}

async def send_typing_action(context, chat_id):
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error(f"Ошибка в typing action: {str(e)}")

async def process_request(func, update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
    typing_task = asyncio.create_task(send_typing_action(context, chat_id))

    try:
        await func(update, context)
    except Exception as e:
        logger.error(f"Ошибка в process_request: {str(e)}")
        if update and update.message:
            await update.message.reply_text("Произошла ошибка при обработке запроса.")
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

def process_document(file_path):
    try:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if os.path.getsize(file_path) == 0:
            raise ValueError("Файл пуст")

        documents = loader.load()
        
        if not documents:
            raise ValueError("Не удалось извлечь содержимое из файла")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.from_documents(texts, embeddings, distance_strategy="COSINE")
        return vector_store

    except Exception as e:
        logger.error(f"Ошибка в process_document: {str(e)}")
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я могу помочь с:\n"
        "1. Отправь любое сообщение текстом, чтобы получить ответ от ИИ\n"
        "2. Используй команду /img и описание, чтобы сгенерировать изображение\n"
        "3. Отправь файлы в форматах PDF, TXT или DOCX для их анализа\n"
        "4. Используй команду /ask и вопрос, чтобы задать вопрос по загруженным документам"
    )

async def ask_document_internal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.message.from_user.id
        question = update.message.text.replace('/ask', '').strip()
        
        if not question:
            await update.message.reply_text("Пожалуйста, добавьте вопрос после команды /ask")
            return

        if user_id not in user_vector_stores:
            await update.message.reply_text("Пожалуйста, сначала загрузите документ.")
            return

        vector_store = user_vector_stores[user_id]
        retriever = vector_store.as_retriever()
        docs = retriever.get_relevant_documents(question)
        context_text = "\n".join([doc.page_content for doc in docs])
        enhanced_prompt = f"Context: {context_text}\n\nQuestion: {question}"

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
                    reply_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'Нет ответа')
                    await update.message.reply_text(reply_text, parse_mode='Markdown')
                else:
                    await update.message.reply_text("Извините, произошла ошибка при обработке запроса.")

    except Exception as e:
        logger.error(f"Ошибка в ask_document: {str(e)}")
        await update.message.reply_text("Произошла ошибка при обработке вопроса.")

async def ask_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_request(ask_document_internal, update, context)

async def handle_text_internal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "messages": [{"role": "user", "content": update.message.text}],
            "max_tokens": 2096,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(CHAT_API_URL, headers=headers, json=data) as response:
                response_json = await response.json()
                if response.status == 200:
                    reply_text = response_json.get('choices', [{}])[0].get('message', {}).get('content', 'Нет ответа')
                    await update.message.reply_text(reply_text, parse_mode='Markdown')
                else:
                    await update.message.reply_text("Извините, произошла ошибка при обработке запроса.")

    except Exception as e:
        logger.error(f"Ошибка в handle_text_internal: {str(e)}")
        await update.message.reply_text("Произошла ошибка при обработке запроса.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_request(handle_text_internal, update, context)

async def handle_document_internal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    temp_file = None
    try:
        await update.message.reply_text("Начинаю обработку документа...")
        
        file = await context.bot.get_file(update.message.document.file_id)
        file_extension = os.path.splitext(update.message.document.file_name)[1].lower()
        
        if file_extension not in ['.pdf', '.txt', '.docx']:
            await update.message.reply_text("Неподдерживаемый формат файла. Пожалуйста, отправьте PDF, TXT или DOCX.")
            return
            
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        await file.download_to_drive(temp_file.name)
        
        if os.path.getsize(temp_file.name) == 0:
            await update.message.reply_text("Файл пуст. Пожалуйста, отправьте файл с содержимым.")
            return

        vector_store = process_document(temp_file.name)
        user_id = update.message.from_user.id
        user_vector_stores[user_id] = vector_store
                
        await update.message.reply_text(
            "Документ успешно обработан! Теперь вы можете задавать вопросы используя команду /ask"
        )
            
    except Exception as e:
        logger.error(f"Ошибка обработки документа: {str(e)}")
        await update.message.reply_text(f"Ошибка обработки документа: {str(e)}")
    
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.error(f"Ошибка при удалении временного файла: {str(e)}")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_request(handle_document_internal, update, context)

async def generate_image_internal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        prompt = update.message.text.replace('/img', '').strip()
        if not prompt:
            await update.message.reply_text("Пожалуйста, добавьте описание после команды /img")
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
                    await update.message.reply_text("Извините, произошла ошибка при генерации изображения.")

    except Exception as e:
        logger.error(f"Ошибка в generate_image_internal: {str(e)}")
        await update.message.reply_text("Произошла ошибка при генерации изображения.")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await process_request(generate_image_internal, update, context)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f'Произошла ошибка: {context.error}')
    if update and update.effective_message:
        await update.effective_message.reply_text('Произошла ошибка. Пожалуйста, попробуйте позже.')

def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Добавление обработчиков
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("img", generate_image))
    application.add_handler(CommandHandler("ask", ask_document))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Добавление обработчика ошибок
    application.add_error_handler(error_handler)

    # Запуск бота
    logger.info("Запуск бота...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()