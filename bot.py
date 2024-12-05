import os
import json
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from contextlib import contextmanager
import sqlite3
import asyncio
import aiohttp

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes, 
    CallbackQueryHandler
)

class Config:
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    load_dotenv()
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

    CHAT_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"
    IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"

    MAX_HISTORY = 10
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_TOKENS = 500
    DB_PATH = "chats.db"

    EMBEDDINGS_CONFIG = {
        "model_name": "nomic-ai/nomic-embed-text-v1.5",
        "model_kwargs": {'device': 'cpu', 'trust_remote_code': True},
        "encode_kwargs": {'normalize_embeddings': True}
    }

class DatabaseManager:
    def __init__(self, db_path: str = Config.DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self.get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (id)
                );

                CREATE TABLE IF NOT EXISTS vector_stores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    vector_data BLOB NOT NULL,
                    created_at TIMESTAMP NOT NULL
                );
            ''')

    def create_chat(self, user_id: int, title: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now()
            cursor.execute(
                "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now)
            )
            return cursor.lastrowid

    def get_user_chats(self, user_id: int, limit: int = 10) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, created_at, updated_at 
                FROM chats 
                WHERE user_id = ? AND is_active = 1
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]

    def save_message(self, chat_id: int, role: str, content: str) -> None:
        with self.get_connection() as conn:
            now = datetime.now()
            conn.execute(
                "INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now)
            )
            conn.execute(
                "UPDATE chats SET updated_at = ? WHERE id = ?",
                (now, chat_id)
            )

    def get_chat_messages(self, chat_id: int, limit: int = Config.MAX_HISTORY) -> List[Dict]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT role, content
                FROM messages 
                WHERE chat_id = ? 
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (chat_id, limit * 2)
            )
            return [dict(row) for row in cursor.fetchall()]

    def save_vector_store(self, user_id: int, vector_store: FAISS) -> None:
        with tempfile.NamedTemporaryFile() as tmp:
            vector_store.save_local(tmp.name)
            with open(tmp.name, 'rb') as f:
                vector_data = f.read()

        with self.get_connection() as conn:
            conn.execute(
                "INSERT INTO vector_stores (user_id, vector_data, created_at) VALUES (?, ?, ?)",
                (user_id, vector_data, datetime.now())
            )

    def get_vector_store(self, user_id: int) -> Optional[FAISS]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT vector_data 
                FROM vector_stores 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
                """,
                (user_id,)
            )
            result = cursor.fetchone()
            
            if result:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(result['vector_data'])
                    tmp.flush()
                    
                    embeddings = HuggingFaceEmbeddings(**Config.EMBEDDINGS_CONFIG)
                    vector_store = FAISS.load_local(tmp.name, embeddings)
                    
                os.unlink(tmp.name)
                return vector_store
            return None

db = DatabaseManager()

class MessageManager:
    @staticmethod
    async def send_typing_action(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        while True:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                await asyncio.sleep(4)
            except asyncio.CancelledError:
                break
            except Exception as e:
                Config.logger.error(f"Ошибка в typing action: {str(e)}")
                break

    @staticmethod
    async def make_api_request(url: str, data: Dict) -> Dict:
        headers = {
            "Authorization": f"Bearer {Config.HUGGINGFACE_TOKEN}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                return await response.json() if response.status == 200 else None

class DocumentProcessor:
    @staticmethod
    def process_document(file_path: str) -> FAISS:
        try:
            loaders = {
                '.pdf': PyPDFLoader,
                '.txt': TextLoader,
                '.docx': Docx2txtLoader
            }
            
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in loaders:
                raise ValueError("Неподдерживаемый формат файла")
            
            loader = loaders[file_ext](file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(**Config.EMBEDDINGS_CONFIG)
            return FAISS.from_documents(texts, embeddings, distance_strategy="COSINE")

        except Exception as e:
            Config.logger.error(f"Ошибка в process_document: {str(e)}")
            raise
class BotHandlers:
    """Класс обработчиков команд и сообщений бота"""

    @staticmethod
    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Привет! Я могу помочь с:\n"
            "1. Отправь любое сообщение текстом, чтобы получить ответ от ИИ\n"
            "2. Используй команду /img и описание, чтобы сгенерировать изображение\n"
            "3. Отправь файлы в форматах PDF, TXT или DOCX для их анализа\n"
            "4. Используй команду /ask и вопрос, чтобы задать вопрос по загруженным документам\n"
            "5. Используй команду /chats для управления чатами\n"
            "6. Используй команду /clear для очистки истории текущего чата"
        )

    @staticmethod
    async def show_chats_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        chats = db.get_user_chats(user_id)
        
        keyboard = [
            [InlineKeyboardButton(f"📝 {chat['title']}", callback_data=f"select_chat_{chat['id']}")]
            for chat in chats
        ]
        keyboard.append([InlineKeyboardButton("➕ Новый чат", callback_data="new_chat")])
        
        await update.message.reply_text(
            "🗂 Ваши чаты:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    @staticmethod
    async def handle_chat_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        
        if not query.data:
            return

        if query.data.startswith("select_chat_"):
            chat_id = int(query.data.replace("select_chat_", ""))
            context.user_data['active_chat'] = chat_id
            
            keyboard = [
                [InlineKeyboardButton("✏️ Переименовать", callback_data=f"rename_{chat_id}")],
                [InlineKeyboardButton("🗑 Удалить", callback_data=f"delete_{chat_id}")],
                [InlineKeyboardButton("◀️ Назад", callback_data="show_chats")]
            ]
            
            await query.edit_message_text(
                "Выбран чат. Что вы хотите сделать?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        elif query.data == "new_chat":
            chat_id = db.create_chat(
                query.from_user.id,
                f"Новый чат {datetime.now().strftime('%d.%m.%Y')}"
            )
            context.user_data['active_chat'] = chat_id
            await query.edit_message_text("Создан новый чат. Можете начинать диалог!")

    @staticmethod
    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        chat_id = update.effective_chat.id if update.effective_chat else update.message.chat_id
        user_id = update.message.from_user.id
        
        typing_task = asyncio.create_task(MessageManager.send_typing_action(context, chat_id))
        
        try:
            active_chat = db.get_user_chats(user_id, limit=1)
            if not active_chat:
                db_chat_id = db.create_chat(
                    user_id,
                    f"Чат от {datetime.now().strftime('%d.%m.%Y')}"
                )
            else:
                db_chat_id = active_chat[0]['id']

            messages = [{"role": "system", "content": "Ты полезный ассистент"}]
            chat_messages = db.get_chat_messages(db_chat_id)
            messages.extend([
                {"role": msg["role"], "content": msg["content"]}
                for msg in reversed(chat_messages)
            ])
            messages.append({"role": "user", "content": update.message.text})

            response = await MessageManager.make_api_request(
                Config.CHAT_API_URL,
                {
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "messages": messages,
                    "max_tokens": Config.MAX_TOKENS,
                    "stream": False
                }
            )

            if response:
                reply_text = response.get('choices', [{}])[0].get('message', {}).get('content', 'Нет ответа')
                await update.message.reply_text(reply_text)
                
                db.save_message(db_chat_id, "user", update.message.text)
                db.save_message(db_chat_id, "assistant", reply_text)
            else:
                await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса.")

        except Exception as e:
            Config.logger.error(f"Ошибка в handle_text: {str(e)}")
            await update.message.reply_text("Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.")
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        chat_id = update.effective_chat.id
        typing_task = asyncio.create_task(MessageManager.send_typing_action(context, chat_id))
        
        try:
            await update.message.reply_text("Начинаю обработку документа...")
            
            file = await context.bot.get_file(update.message.document.file_id)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(update.message.document.file_name)[1]) as tmp_file:
                await file.download_to_drive(tmp_file.name)
                vector_store = DocumentProcessor.process_document(tmp_file.name)
                db.save_vector_store(update.message.from_user.id, vector_store)
                os.unlink(tmp_file.name)
                
            await update.message.reply_text(
                "Документ успешно обработан! Теперь вы можете задавать вопросы по его содержимому с помощью команды /ask"
            )
            
        except Exception as e:
            Config.logger.error(f"Ошибка обработки документа: {str(e)}")
            await update.message.reply_text(f"Ошибка обработки документа: {str(e)}")
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def ask_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        user_id = update.message.from_user.id
        chat_id = update.effective_chat.id
        
        question = update.message.text.replace('/ask', '').strip()
        if not question:
            await update.message.reply_text("Пожалуйста, добавьте вопрос после команды /ask")
            return

        vector_store = db.get_vector_store(user_id)
        if not vector_store:
            await update.message.reply_text("Пожалуйста, сначала загрузите документ.")
            return

        typing_task = asyncio.create_task(MessageManager.send_typing_action(context, chat_id))
        
        try:
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context_text = "\n".join([doc.page_content for doc in docs])
            
            messages = [
                {"role": "system", "content": "Ты полезный ассистент"},
                {"role": "user", "content": f"Контекст: {context_text}\n\nВопрос: {question}"}
            ]
            
            response = await MessageManager.make_api_request(
                Config.CHAT_API_URL,
                {
                    "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                    "messages": messages,
                    "max_tokens": Config.MAX_TOKENS,
                    "stream": False
                }
            )

            if response:
                reply_text = response.get('choices', [{}])[0].get('message', {}).get('content', 'Нет ответа')
                await update.message.reply_text(reply_text)
            else:
                await update.message.reply_text("Извините, произошла ошибка при обработке вашего запроса.")

        except Exception as e:
            Config.logger.error(f"Ошибка в ask_document: {str(e)}")
            await update.message.reply_text("Произошла ошибка при обработке вашего вопроса.")
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        chat_id = update.effective_chat.id
        prompt = update.message.text.replace('/img', '').strip()
        
        if not prompt:
            await update.message.reply_text("Пожалуйста, добавьте описание после команды /img")
            return

        typing_task = asyncio.create_task(MessageManager.send_typing_action(context, chat_id))
        
        try:
            response = await MessageManager.make_api_request(
                Config.IMAGE_API_URL,
                {"inputs": prompt}
            )
            
            if response:
                await update.message.reply_photo(response)
            else:
                await update.message.reply_text("Извините, произошла ошибка при генерации изображения.")
                
        except Exception as e:
            Config.logger.error(f"Ошибка в generate_image: {str(e)}")
            await update.message.reply_text("Произошла ошибка при генерации изображения.")
        finally:
            if typing_task and not typing_task.done():
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def clear_context(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.message.from_user.id
            new_chat_id = db.create_chat(
                user_id,
                f"Новый чат {datetime.now().strftime('%d.%m.%Y')}"
            )
            context.user_data['active_chat'] = new_chat_id
            await update.message.reply_text("Контекст очищен! Создан новый чат.")
        except Exception as e:
            Config.logger.error(f"Ошибка при очистке контекста: {str(e)}")
            await update.message.reply_text("Произошла ошибка при очистке контекста.")

    @staticmethod
    async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        Config.logger.error(f'Произошла ошибка: {context.error}')
        if update and update.effective_message:
            await update.effective_message.reply_text(
                'Произошла ошибка. Пожалуйста, попробуйте позже.'
            )

class TelegramBot:
    def __init__(self):
        self.application = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        handlers = [
            CommandHandler("start", BotHandlers.start),
            CommandHandler("img", BotHandlers.generate_image),
            CommandHandler("ask", BotHandlers.ask_document),
            CommandHandler("clear", BotHandlers.clear_context),
            CommandHandler("chats", BotHandlers.show_chats_menu),
            CallbackQueryHandler(BotHandlers.handle_chat_selection),
            MessageHandler(filters.Document.ALL, BotHandlers.handle_document),
            MessageHandler(filters.TEXT & ~filters.COMMAND, BotHandlers.handle_text)
        ]
        
        for handler in handlers:
            self.application.add_handler(handler)

        self.application.add_error_handler(BotHandlers.error_handler)

    def run(self) -> None:
        Config.logger.info("Запуск бота...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main() -> None:
    try:
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        Config.logger.error(f"Критическая ошибка при запуске бота: {str(e)}")

if __name__ == '__main__':
    main()