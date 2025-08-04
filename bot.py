import os
import aiohttp
import disnake
from disnake.ext import commands
from disnake import Option, OptionType
import openai
from openai import OpenAI
from enum import Enum
import asyncio
import io
import docx
import database
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import disnake.ui as ui
import base64
import mimetypes
from typing import Optional

# --- Configuration ---
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK = os.getenv("DEEPSEEK_API")
GROK = os.getenv("GROK_API")
GROQ = os.getenv("GROQ_API")
GOOGLE_API = os.getenv("GOOGLE_API")
GLM = os.getenv("GLM_API")

MODELS = {
    "openai": {
        "GPT 4.1 Mini": "gpt-4.1-mini",
        "GPT 4.1": "gpt-4.1",
        "o3": "o3",
        "o4 mini" : "o4-mini"
    },
    "deepseek": {
        "DeepSeek V3": "deepseek-chat",
        "DeepSeek R1": "deepseek-reasoner"
    },
    "grok": {
        "Grok 4": "grok-4-0709"
    },
    "google": {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 2.5 Pro": "gemini-2.5-pro"
    },
    "groq": {
        "Compound Beta": "compound-beta",
        "Kimi K2": "moonshotai/kimi-k2-instruct",
        "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
        "Qwen3 32B": "qwen/qwen3-32b",
        "Llama-4 Maverick": "meta-llama/llama-4-maverick-17b-128e-instruct"
    },
    "glm": {
        "GLM 4.5": "GLM-4.5",
        "GLM 4.5 X": "GLM-4.5-X"
    }
}

intents = disnake.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", help_command=None, intents=intents)

def det_provider(selected_provider):
    providers_data = {}
    if selected_provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = OPENAI_API_KEY

    elif selected_provider == "deepseek":
        base_url = "https://api.deepseek.com"
        api_key = DEEPSEEK

    elif selected_provider == "grok":
        base_url = "https://api.x.ai/v1"
        api_key = GROK

    elif selected_provider == "google":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = GOOGLE_API

    elif selected_provider == "groq":
        base_url = "https://api.groq.com/openai/v1"
        api_key = GROQ
    
    elif selected_provider == "glm":
        base_url = "https://api.z.ai/api/paas/v4"
        api_key = GLM
    providers_data["base_url"] = base_url
    providers_data["api_key"] = api_key
    return providers_data


class MyView(ui.View):
    def __init__(self, channel_id: str):
        super().__init__(timeout=None)
        self.bot = bot
        self.channel_id = channel_id
        i = 0
        for names in MODELS.values():
            for name in names.keys():
                row = i//4
                btn = ui.Button(label=name, style=disnake.ButtonStyle.primary, row=row)
                btn.callback = self.func_button
                self.add_item(btn)
                i += 1
        i = 0
    async def func_button(self, interaction: disnake.MessageInteraction):
        label_id = interaction.component.label
        
        for provider, models in MODELS.items():
            for model in models.keys():
                if model == label_id:
                    selected_provider = provider
                    model_name = models[model]
                    database.update_model(self.channel_id, model_name)

        provider_data = det_provider(selected_provider)
        # Обновляем настройки API в базе
        database.update_api_settings(self.channel_id, selected_provider, provider_data["base_url"], provider_data["api_key"])

        await interaction.response.send_message(
            f"Модель: {model_name}",
        )


# Get the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
CONTEXT_FILE = os.path.join(script_dir, "контекст.txt")
HELP_FILE = os.path.join(script_dir, "help.txt")
IMAGE_DIR = os.path.join(script_dir, "Картинки")

# --- Initialization ---
database.init_db()

if not DISCORD_TOKEN:
    raise ValueError("Discord token not found. Please set the DISCORD_TOKEN environment variable.")






try:
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        initial_role_prompt = f.read()
except FileNotFoundError:
    initial_role_prompt = "You are a helpful assistant."
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write(initial_role_prompt)





stop_processing = False

# --- Helper Functions ---
async def extract_txt(file_content):
    return file_content.decode('utf-8')

async def extract_docx(file_content):
    doc = docx.Document(io.BytesIO(file_content))
    return "\n".join([p.text for p in doc.paragraphs])

async def extract_html(file_content):
    soup = BeautifulSoup(file_content, 'lxml')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

async def url_to_base64_data_uri(
    session: aiohttp.ClientSession, 
    url: str
) -> Optional[str]:
    """
    Асинхронно скачивает изображение по URL и кодирует его в Base64 Data URI.

    :param session: Активная сессия aiohttp.ClientSession для выполнения запроса.
    :param url: URL-адрес изображения.
    :return: Строку Data URI или None в случае ошибки.
    """
    try:
        async with session.get(url) as response:
            # Проверяем, что запрос прошел успешно
            if response.status != 200:
                print(f"Ошибка: не удалось скачать изображение. Статус: {response.status}")
                return None

            # Читаем байты изображения
            image_bytes = await response.read()
            
            # Получаем MIME-тип из заголовков (например, 'image/jpeg')
            mime_type = response.headers.get('Content-Type', 'application/octet-stream')

            # Кодируем байты в строку base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')

            # Формируем и возвращаем готовый Data URI
            return f"data:{mime_type};base64,{base64_string}"
            
    except Exception as e:
        print(f"Произошла ошибка при обработке URL {url}: {e}")
        return None

async def handle_attachments(message, channel_data, content_list = None):
    
    model_name = channel_data[2]
    # Gemini models require base64 encoding. Other vision models like GPT-4o can handle URLs.
    is_google_model = 'gemini' in model_name.lower()
    for attachment in message.attachments:
        if attachment.content_type and attachment.content_type.startswith("image/"):
            content_list.append({"type": "image_url", "image_url": {"url": attachment.url}})

        elif attachment.filename.endswith(('.txt', '.docx', '.html')):
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status == 200:
                        file_content = await resp.read()
                        if attachment.filename.endswith('.txt'):
                            extracted_text = await extract_txt(file_content)
                        elif attachment.filename.endswith('.docx'):
                            extracted_text = await extract_docx(file_content)
                        elif attachment.filename.endswith('.html'):
                            extracted_text = await extract_html(file_content)

                        if extracted_text:
                            
                            content_list[0]["text"] += f"\n\n--- Content from {attachment.filename} ---\n{extracted_text}"

async def send_reply(channel, text):
    if len(text) > 2000:
        parts = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for part in parts:
            await channel.send(part)
    else:
        await channel.send(text)

# --- Bot Events ---
@bot.event
async def on_guild_join(guild):
    print(f"Бот добавлен на сервер {guild.name}")
    if guild.system_channel:
        await guild.system_channel.send("Благодарю за приглашение в ваш клуб господа!\nПрошу документация бота и список команд")

@bot.event
async def on_ready():
    print(f"{bot.user} now online!")

@bot.event
async def on_message(message):
    if message.author == bot.user or stop_processing:
        return

    # Check if it's a command
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    channel_id = str(message.channel.id)
    if not database.get_channel(channel_id):
        return

    user_message_text = f"{message.author.display_name}: {message.content}"
    content_list = [{"type": "text", "text": user_message_text}]

    channel_data = database.get_channel(channel_id)

    if message.attachments:
        await handle_attachments(message, channel_data, content_list)

    database.add_message(channel_id, "user", content_list)

    messages = database.get_messages(channel_id)

    time_history = list(messages)
    
    if "gemini" in channel_data[0].lower():
        async with aiohttp.ClientSession() as session:
            for msg in time_history:
                if isinstance(msg["content"], list):
                    for item in msg["content"]:
                        if item["type"] == "image_url" and item["image_url"]["url"].startswith("http"):
                            base64_uri = await url_to_base64_data_uri(session, item["image_url"]["url"])
                            if base64_uri:
                                item["image_url"]["url"] = base64_uri

    base_url_from_db = channel_data[3]
    api_key_from_db = channel_data[4]

    client = OpenAI(base_url = base_url_from_db, api_key = api_key_from_db)
    try:
        response = client.chat.completions.create(
            model=channel_data[0],
            messages=time_history
        )
        reply = response.choices[0].message.content
        await send_reply(message.channel, reply)
        database.add_message(channel_id, "assistant", reply)
    except openai.APIError as e:
        await message.channel.send(f"Возникла ошибка OpenAI API: {e}")
    except Exception as e:
        await message.channel.send(f"Возникла непредвиденная ошибка: {e}")

# --- Slash Commands ---
@bot.slash_command(description="Очищает память для текущего чата")
async def forget(inter):
    channel_id = str(inter.channel.id)
    if not database.get_channel(channel_id):
        await inter.response.send_message("Данный канал не установлен.", ephemeral=True)
        return
    database.clear_history(channel_id)
    await inter.response.send_message("Память чата очищена.", ephemeral=True)

@bot.slash_command(description="Устанавливает текущий канал для общения")
async def setup(inter):
    channel_id = str(inter.channel.id)
    if database.get_channel(channel_id):
        await inter.response.send_message("Данный канал уже установлен.", ephemeral=True)
    else:
        database.add_channel(
            channel_id, 
            inter.channel.name, 
            "gpt-4.1-mini", 
            initial_role_prompt, 
            "openai", # Дефолтный провайдер
            "https://api.openai.com/v1", # Дефолтный URL
            OPENAI_API_KEY
        )
        database.add_message(channel_id, "system", initial_role_prompt)
        await inter.response.send_message(f"Канал '{inter.channel.name}' установлен для общения!")

@bot.slash_command(description="Показывает справку по командам")
async def help(inter):
    try:
        with open(HELP_FILE, "r", encoding="utf-8") as file:
            help_text = file.read()
        embed = disnake.Embed(title="Справка по командам", description=help_text, color=0x00FF00)
        await inter.response.send_message(embed=embed, ephemeral=True)
    except FileNotFoundError:
        await inter.response.send_message("Файл справки не найден.", ephemeral=True)

@bot.slash_command(description="Удаляет текущий канал для общения")
async def del_channel(inter):
    channel_id = str(inter.channel.id)
    if database.get_channel(channel_id):
        database.delete_channel(channel_id)
        await inter.response.send_message(f"Канал '{inter.channel.name}' успешно удален из списка отслеживаемых.")
    else:
        channels = database.get_all_channels()
        channel_list = "\n".join([name for id, name in channels]) if channels else "Нет установленных каналов."
        await inter.response.send_message(f"Этот канал не отслеживается.\n**Установленные каналы:**\n{channel_list}", ephemeral=True)

@bot.slash_command(description="Остановит общение в чате")
async def stop(inter):
    global stop_processing
    stop_processing = True
    await inter.response.send_message("Общение остановлено.", ephemeral=True)

@bot.slash_command(description="Возобновит остановленное общение")
async def resume(inter):
    global stop_processing
    stop_processing = False
    await inter.response.send_message("Общение возобновлено.", ephemeral=True)

@bot.slash_command(description="Отправляет информацию о чате")
async def info(inter):
    channel_id = str(inter.channel.id)
    channel_data = database.get_channel(channel_id)
    if not channel_data:
        await inter.response.send_message("Ошибка: канал не установлен.", ephemeral=True)
        return
    embed = disnake.Embed(
        title="Информация о чате",
        description=f"**Канал:** {inter.channel.name}\n**Модель:** {channel_data[0]}\n**Контекст:** {channel_data[1]}",
        color=0x00FF00
    )
    await inter.response.send_message(embed=embed, ephemeral=True)

@bot.slash_command(description="Обновляет контекст текущему чату")
async def new_con(inter, context: str):
    channel_id = str(inter.channel.id)
    if not database.get_channel(channel_id):
        await inter.response.send_message("Нельзя обновить контекст в неустановленном канале!", ephemeral=True)
        return
    database.update_context(channel_id, context)
    await inter.response.send_message("Контекст успешно обновлен.", ephemeral=True)

@bot.slash_command(description="Выбрать модель для ответов")
async def model(inter):
    channel_id = str(inter.channel.id)
    if database.get_channel(channel_id) == None:
        await inter.response.send_message("Данный канал не установлен.", ephemeral=True)
        return
    await inter.channel.send("*Выбор модели*", view = MyView(channel_id = channel_id))

@bot.slash_command(description="Удаляет указанное кол-во сообщений")
async def clear(inter, count: int):
    if count <= 0:
        await inter.response.send_message("Укажите положительное число сообщений для удаления.", ephemeral=True)
        return
    deleted = await inter.channel.purge(limit=count)
    await inter.response.send_message(f"Удалено {len(deleted)} сообщений.", delete_after=5, ephemeral=True)

@bot.slash_command(description="Сбрасывает контекст на стандартный")
async def back_con(inter):
    channel_id = str(inter.channel.id)
    if not database.get_channel(channel_id):
        await inter.response.send_message("Нельзя сбросить контекст в неустановленном канале!", ephemeral=True)
        return
    database.update_context(channel_id, initial_role_prompt)
    await inter.response.send_message("Контекст сброшен на стандартный.", ephemeral=True)

@bot.slash_command(description="Создает новый чат для общения с ботом")
async def new_chat(inter, name: str = ""):
    guild = inter.guild
    if not name:
        i = 1
        while True:
            name = f"{inter.user.name}-чат-{i}"
            if not disnake.utils.get(guild.text_channels, name=name):
                break
            i += 1
    
    try:
        channel = await guild.create_text_channel(name)
        channel_id = str(channel.id)
        database.add_channel(channel_id, channel.name, "gpt-4.1-mini", initial_role_prompt, "openai", "https://api.openai.com/v1", OPENAI_API_KEY)
        database.add_message(channel_id, "system", initial_role_prompt)
        await inter.response.send_message(f"Канал '{channel.name}' создан! -> <#{channel.id}>")
    except disnake.Forbidden:
        await inter.response.send_message("У меня нет прав для создания каналов.", ephemeral=True)
    except Exception as e:
        await inter.response.send_message(f"Произошла ошибка при создании канала: {e}", ephemeral=True)

# --- Image Generation ---
Size = commands.option_enum(["1024x1024", "1024x1792", "1792x1024"])
Quality = commands.option_enum(["hd", "standard"])
Model = commands.option_enum(["dall-e-2", "dall-e-3"])

@bot.slash_command(description="Генерирует изображение по текстовому описанию")
async def gen(inter: disnake.ApplicationCommandInteraction, prompt: str, size: Size = "1024x1024", model: Model = "dall-e-3", quality: Quality = "standard", count: int = 1):
    if model == 'dall-e-3' and count > 1:
        await inter.response.send_message("Модель DALL-E 3 поддерживает только 1 генерацию за раз.", ephemeral=True)
        return
    if model == "dall-e-2" and size != "1024x1024":
        await inter.response.send_message("Модель DALL-E 2 поддерживает только размер 1024x1024.", ephemeral=True)
        return
    
    await inter.response.defer(ephemeral=True)
    
    try:
        # The 'client' object was not defined in this scope.
        # We need to create an OpenAI client here to use the images API.
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        image_response = openai_client.images.generate(
            model=model,
            prompt=prompt,
            n=count,
            size=size,
            quality=quality
        )
        
        await inter.followup.send("Изображения сгенерированы. Отправляю в канал...")
        
        for i, image_data in enumerate(image_response.data):
            revised_prompt = image_data.revised_prompt or prompt
            embed = disnake.Embed(
                title=f"Изображение {i+1}/{count}",
                description=f"**Prompt:** {revised_prompt}",
                color=0x00FF00
            )
            embed.set_image(url=image_data.url)
            embed.set_footer(text=f"Автор: {inter.user.display_name} | Модель: {model}")
            
            await inter.channel.send(embed=embed)

    except openai.APIError as e:
        await inter.followup.send(f"Ошибка OpenAI API: {e}", ephemeral=True)
    except Exception as e:
        await inter.followup.send(f"Произошла непредвиденная ошибка: {e}", ephemeral=True)

Model_ask = commands.option_enum(["gpt-4.1", "grok-4-0709", "gemini-2.5-pro", "gpt-4.1-mini", "gemini-2.5-flash"])
@bot.slash_command(description="Быстро спросить у бота в любом чате по последним сообщениям")
async def ask(inter, prompt: str, context: int = 0, model: Model_ask = "gemini-2.5-flash"):
    await inter.response.defer()
    zapros = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
    
    for name in MODELS.keys():
        if model in MODELS[name].values():
            provider = name
            break

    async for msg in inter.channel.history(limit = context, before=inter):
        zapros[0]["content"].append({"type": "text", "text": msg.content})
        if msg.attachments:
            for attachment in msg.attachments:
                if attachment.content_type and attachment.content_type.startswith("image/") and attachment.content_type != 'image/gif':
                    if provider == "google":
                        async with aiohttp.ClientSession() as session:
                            data_url = await url_to_base64_data_uri(session, attachment.url)
                            zapros[0]["content"].append({"type": "image_url", "image_url": {"url": data_url}})
                    else:
                        zapros[0]["content"].append({"type": "image_url", "image_url": {"url": attachment.url}})
                elif attachment.filename.endswith(('.txt', '.docx', '.html')):
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                file_content = await resp.read()
                                extracted_text = ""
                                if attachment.filename.endswith('.txt'):
                                    extracted_text = await extract_txt(file_content)
                                elif attachment.filename.endswith('.docx'):
                                    extracted_text = await extract_docx(file_content)
                                elif attachment.filename.endswith('.html'):
                                    extracted_text = await extract_html(file_content)

                                if extracted_text:
                                    zapros[0]["content"].append({"type": "text", "text": f"\n\n--- Content from {attachment.filename} ---\n{extracted_text}"})

    client_info = det_provider(provider)
    ask_quiestion = OpenAI(api_key = client_info["api_key"], base_url = client_info["base_url"])

    print(zapros)
    try:
        completion = ask_quiestion.chat.completions.create(
            model = model,
            messages = zapros
            )
        answer = completion.choices[0].message.content
        print(answer)
        if len(answer) > 2000:
            parts = [answer[i: i + 2000] for i in range(0, len(answer), 2000)]
            for part in parts:
                await inter.followup.send(part)
        else:
            await inter.followup.send(answer)
    except openai.APIError as e:
        await inter.followup.send(f"Ошибка OpenAI: {e}")
    except Exception as e:
        await inter.followup.send(f"Ошибка: {e}")

# --- Text Commands (for owner) ---
@bot.command()
@commands.is_owner()
async def отключить(ctx):
    await ctx.send("Бот отключается...")
    await bot.close()

@bot.command()
async def каналы(ctx):
    channels = database.get_all_channels()
    if channels:
        channel_list = "\n".join([name for id, name in channels])
        await ctx.send(f"**Установленные каналы:**\n{channel_list}")
    else:
        await ctx.send("Установленных каналов нет.")

# --- Main Execution ---
if __name__ == "__main__":
    if DISCORD_TOKEN:
        bot.run(DISCORD_TOKEN)
    else:
        print("Ошибка: DISCORD_TOKEN не найден в .env файле.")