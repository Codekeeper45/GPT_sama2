
import sqlite3
import json

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS channels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT UNIQUE NOT NULL,
            channel_name TEXT NOT NULL,
            model TEXT,
            system_prompt TEXT,
            provider TEXT,
            base_url TEXT,
            api_key TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
        )
    """)
    conn.commit()
    conn.close()

def add_channel(channel_id, channel_name, model, system_prompt, provider, base_url, api_key):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO channels (channel_id, channel_name, model, system_prompt, provider, base_url, api_key) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (channel_id, channel_name, model, system_prompt, provider, base_url, api_key))
        conn.commit()
    except sqlite3.IntegrityError:
        # Channel already exists
        pass
    finally:
        conn.close()

def delete_channel(channel_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM channels WHERE channel_id = ?", (channel_id,))
    c.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,))
    conn.commit()
    conn.close()

def get_channel(channel_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT model, system_prompt, provider, base_url, api_key FROM channels WHERE channel_id = ?", (channel_id,))
    channel_data = c.fetchone()
    conn.close()
    return channel_data

def get_all_channels():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT channel_id, channel_name FROM channels")
    channels = c.fetchall()
    conn.close()
    return channels

def add_message(channel_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    c.execute("INSERT INTO messages (channel_id, role, content) VALUES (?, ?, ?)", (channel_id, role, content))
    conn.commit()
    conn.close()

def get_messages(channel_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE channel_id = ?", (channel_id,))
    messages = []
    for role, content in c.fetchall():
        try:
            parsed_content = json.loads(content)
            messages.append({"role": role, "content": parsed_content})
        except (json.JSONDecodeError, TypeError):
            messages.append({"role": role, "content": content})
    conn.close()
    return messages

def clear_history(channel_id):
    conn = sqlite3.connect('chat_history.db', timeout=10)
    c = conn.cursor()
    # Get the system prompt before deleting messages
    c.execute("SELECT system_prompt FROM channels WHERE channel_id = ?", (channel_id,))
    system_prompt_row = c.fetchone()
    if system_prompt_row:
        system_prompt = system_prompt_row[0]
        # Delete all messages except the system prompt
        c.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,))
        # Add the system prompt back
        c.execute("INSERT INTO messages (channel_id, role, content) VALUES (?, ?, ?)", (channel_id, "system", system_prompt))
    conn.commit()
    conn.close()


def update_context(channel_id, new_context):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("UPDATE channels SET system_prompt = ? WHERE channel_id = ?", (new_context, channel_id))
    # Also update the system message in the messages table
    c.execute("UPDATE messages SET content = ? WHERE channel_id = ? AND role = 'system'", (new_context, channel_id))
    conn.commit()
    conn.close()

def update_model(channel_id, new_model):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("UPDATE channels SET model = ? WHERE channel_id = ?", (new_model, channel_id))
    conn.commit()
    conn.close()

def update_api_settings(channel_id, provider, base_url, api_key):
    conn = sqlite3.connect('chat_history.db', timeout=10)
    c = conn.cursor()
    c.execute(
        "UPDATE channels SET provider = ?, base_url = ?, api_key = ? WHERE channel_id = ?",
        (provider, base_url, api_key, channel_id)
    )
    conn.commit()
    conn.close()