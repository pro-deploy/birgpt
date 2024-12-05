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

def process_document(file_path):
    try:
        # Определение загрузчика на основе расширения файла
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Неподдерживаемый формат файла")

        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={
                'device': 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True
            }
        )
        
        vector_store = FAISS.from_documents(
            texts, 
            embeddings,
            distance_strategy="COSINE"
        )
        
        return vector_store

    except Exception as e:
        logger.error(f"Ошибка в process_document: {str(e)}")
        raise

async def send_typing_action(context, chat_id):
    while True:
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Ошибка в typing action: {str(e)}")
            break

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я могу помочь с:\n"
        "1. Отправь любое сообщение текстом, чтобы получить ответ от ИИ\n"
        "2. Используй команду /img и описание, чтобы сгенерировать изображение\n"
        "3. Отправь файлы в форматах PDF, TXT или DOCX для их анализа\n"
        "4. Используй команду /ask и вопрос, чтобы задать вопрос по загруженным документам"
    )

async def ask_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message:
            logger.error("Нет сообщения в обновлении")
            return

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
        
        if not chat_id:
            logger.error("Нет доступного chat_id")
            return

        question = update.message.text.replace('/ask', '').strip()
        if not question:
            await update.message.reply_text("Пожалуйста, добавьте вопрос после команды /ask")
            return

        if user_id not in user_vector_stores:
            await update.message.reply_text("Пожалуйста, сначала загрузите документ, прежде чем задавать вопросы.")
            return

        typing_task = asyncio.create_task(send_typing_action(context, chat_id))
        
        try:
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
                        await update.message.reply_text(reply_text)
                    else:
                        await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса.")
        
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Ошибка в ask_document: {str(e)}")
        if update and update.message:
            await update.message.reply_text("Произошла непредвиденная ошибка при обработке вашего вопроса.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message:
            logger.error("Нет сообщения в обновлении")
            return

        chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
        
        if not chat_id:
            logger.error("Нет доступного chat_id")
            return

        typing_task = asyncio.create_task(send_typing_action(context, chat_id))
        
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
                        await update.message.reply_text(reply_text)
                    else:
                        await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса.")
        
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Критическая ошибка в handle_text: {str(e)}")
        if update and update.message:
            await update.message.reply_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    typing_task = None
    try:
        if not update.message:
            logger.error("Нет сообщения в обновлении")
            return

        chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
        
        if not chat_id:
            logger.error("Нет доступного chat_id")
            return

        # Начинаем показывать "печатает" сразу при получении документа
        typing_task = asyncio.create_task(send_typing_action(context, chat_id))
        
        await update.message.reply_text("Начинаю обработку документа...")
        
        try:
            file = await context.bot.get_file(update.message.document.file_id)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(update.message.document.file_name)[1]) as tmp_file:
                await file.download_to_drive(tmp_file.name)
                
                vector_store = process_document(tmp_file.name)
                
                user_id = update.message.from_user.id
                user_vector_stores[user_id] = vector_store
                
                os.unlink(tmp_file.name)
                
            await update.message.reply_text(
                "Документ успешно обработан! Теперь вы можете задавать вопросы по его содержимому с помощью команды /ask"
            )
            
        except Exception as e:
            logger.error(f"Ошибка обработки документа: {str(e)}")
            await update.message.reply_text(f"Ошибка обработки документа: {str(e)}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка в handle_document: {str(e)}")
        if update and update.message:
            await update.message.reply_text("Произошла непредвиденная ошибка при обработке документа.")
    finally:
        if typing_task and not typing_task.done():
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not update.message:
            logger.error("Нет сообщения в обновлении")
            return

        chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
        
        if not chat_id:
            logger.error("Нет доступного chat_id")
            return

        typing_task = asyncio.create_task(send_typing_action(context, chat_id))
        
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
        
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Критическая ошибка в generate_image: {str(e)}")
        if update and update.message:
            await update.message.reply_text("Произошла непредвиденная ошибка при генерации изображения.")

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