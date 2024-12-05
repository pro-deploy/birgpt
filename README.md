# AI Chat & Image Generation Telegram Bot with Document Analysis

Бот с возможностями:
1. Отвечать на текстовые сообщения используя AI
2. Генерировать изображения по описанию (команда `/img`)
3. Анализировать документы (PDF, DOCX, TXT) и отвечать на вопросы по их содержимому

## Предварительные требования

### 1. Установка Docker и Docker Compose

#### Windows:
1. Скачайте и установите [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Docker Compose уже включен в установку

#### Ubuntu:
```bash
# Установка Docker
sudo apt-get update
sudo apt-get install docker.io

# Установка Docker Compose
sudo apt-get install curl
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Запуск Docker
sudo systemctl start docker
sudo systemctl enable docker

# Добавление вашего пользователя в группу docker
sudo usermod -aG docker $USER
```
⚠️ После этого перезагрузите систему или выполните:
```bash
newgrp docker
```

#### Mac:
1. Скачайте и установите [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Docker Compose уже включен в установку

### 2. Получение токенов

#### Telegram Bot Token:
1. Найдите @BotFather в Telegram
2. Отправьте `/newbot`
3. Следуйте инструкциям:
   - Введите имя бота
   - Введите username (должен заканчиваться на 'bot')
4. Сохраните полученный токен

#### Hugging Face Token:
1. Зарегистрируйтесь на [Hugging Face](https://huggingface.co/)
2. Перейдите в [Settings -> Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте новый токен (New token)
4. Выберите права "read"
5. Сохраните полученный токен

## Установка бота

1. Клонируйте репозиторий:
```bash
git clone https://github.com/pro-deploy/birgpt.git
cd https://github.com/pro-deploy/birgpt.git
```

2. Создайте файл с настройками:
```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

3. Отредактируйте `.env`:
```
TELEGRAM_BOT_TOKEN=ваш_телеграм_токен
HUGGINGFACE_TOKEN=ваш_huggingface_токен
```

4. Запустите бота:
```bash
docker-compose up --build
```

## Использование бота

### Базовые функции:
1. `/start` - начало работы
2. Отправка текстовых сообщений → получение ответов от AI
3. `/img описание` → генерация изображения

### Работа с документами:
1. Отправьте файл (PDF, DOCX или TXT) в чат
2. Дождитесь сообщения об успешной обработке
3. Задавайте вопросы по содержимому документа

Примеры:
- Простой чат: "Что такое машинное обучение?"
- Генерация изображения: `/img cat playing piano`
- Анализ документа: 
  1. Отправьте PDF файл
  2. Спросите "Какие основные тезисы в документе?"

## Устранение проблем

### Проблема: Docker не запускается
- Убедитесь, что Docker Desktop запущен (для Windows/Mac)
- Для Linux проверьте статус: `sudo systemctl status docker`

### Проблема: Бот не отвечает
1. Проверьте правильность токенов в файле `.env`
2. Убедитесь, что бот запущен (в терминале должны быть логи)
3. Попробуйте перезапустить бота:
```bash
docker-compose down
docker-compose up --build
```

### Проблема: Ошибки при сборке
1. Убедитесь, что все файлы на месте
2. Проверьте подключение к интернету
3. Попробуйте очистить Docker кэш:
```bash
docker system prune -a
```

### Ошибка доступа к Docker
```bash
sudo chmod 666 /var/run/docker.sock
```

### Документ не обрабатывается
- Проверьте формат (поддерживаются PDF, DOCX, TXT)
- Убедитесь, что файл не поврежден
- Проверьте размер файла (не более 20MB для Telegram)


## Ограничения
- Векторное хранилище хранится в памяти
- При перезапуске бота загруженные документы нужно загружать заново
- Максимальный размер файла: 20MB (ограничение Telegram)

## Для разработчиков

### Локальный запуск без Docker:
```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Запуск бота
python bot.py
```

### Требования к системе:
- Python 3.9+
- 6GB RAM минимум
- Для работы с PDF требуется poppler-utils

## Безопасность
- Не публикуйте файл `.env`
- Регулярно обновляйте токены
- Не загружайте конфиденциальные документы

## Поддержка
- Создавайте Issues в репозитории
- Проверяйте раздел Troubleshooting
- При сообщении об ошибках прикладывайте логи (без токенов!)