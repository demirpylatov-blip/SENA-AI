import os
import tempfile
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# =========================
# ENV
# =========================
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# UI
# =========================
KEYBOARD = ReplyKeyboardMarkup(
    [
        ["🎯 IELTS", "💬 General English"],
        ["🟢 Part 1", "🟡 Part 2", "🔵 Part 3"],
        ["✅ Finish", "🔄 Reset"],
        ["ℹ️ Help"],
    ],
    resize_keyboard=True,
)

# =========================
# MEMORY
# =========================
# In-memory session storage
# Works fine for one Render worker.
sessions: Dict[int, Dict] = {}


def default_session() -> Dict:
    return {
        "name": None,
        "awaiting_name": True,
        "mode": None,          # "general" / "ielts"
        "part": None,          # "part1" / "part2" / "part3"
        "history": [],         # list of {"role": "...", "content": "..."}
        "started": False,
    }


def get_session(user_id: int) -> Dict:
    if user_id not in sessions:
        sessions[user_id] = default_session()
    return sessions[user_id]


def reset_session(user_id: int) -> Dict:
    sessions[user_id] = default_session()
    return sessions[user_id]


# =========================
# PROMPTS
# =========================
GENERAL_SYSTEM = """
You are Sena AI, a smart English speaking coach.

Style:
- natural
- warm
- short
- useful
- not robotic

Rules:
- help the student improve spoken English
- correct mistakes naturally
- reply in a simple, human way
- if the user asks for practice, ask ONE useful speaking question
- if the user's sentence is weak, improve it briefly
- do not give long lectures
"""

IELTS_PART1_SYSTEM = """
You are Sena AI in IELTS Speaking Part 1 mode.

Rules:
- Ask only ONE IELTS Part 1 question at a time.
- Keep it natural and exam-like.
- React briefly to the student's answer.
- Then ask the next question.
- Do not give full final feedback during the session.
"""

IELTS_PART2_SYSTEM = """
You are Sena AI in IELTS Speaking Part 2 mode.

Rules:
- Give a cue card or one short follow-up at a time.
- Encourage the student to speak in detail.
- Keep replies short.
- Do not score during the session.
"""

IELTS_PART3_SYSTEM = """
You are Sena AI in IELTS Speaking Part 3 mode.

Rules:
- Ask only ONE deeper discussion question at a time.
- Keep it concise and natural.
- React briefly, then continue.
- Do not score during the session.
"""

GENERAL_FEEDBACK_SYSTEM = """
You are Sena AI.
Give final speaking feedback for a general English speaking session.

Return exactly these sections:
Name:
Mode:
Overall feedback:
Strong points:
Mistakes to improve:
Better examples:
Next advice:

Keep it short, natural, and useful.
"""

IELTS_FEEDBACK_SYSTEM = """
You are Sena AI.
Give final IELTS-style speaking feedback.

Return exactly these sections:
Name:
Mode:
Estimated band:
Why:
Grammar:
Vocabulary:
Fluency:
Pronunciation:
Better versions:
Advice for improvement:

Rules:
- Band must be from 1 to 9
- Be realistic
- Keep it useful and not too long
"""

# =========================
# HELPERS
# =========================
def current_system_prompt(state: Dict) -> str:
    if state["mode"] == "ielts":
        if state["part"] == "part1":
            return IELTS_PART1_SYSTEM
        if state["part"] == "part2":
            return IELTS_PART2_SYSTEM
        if state["part"] == "part3":
            return IELTS_PART3_SYSTEM
        return IELTS_PART1_SYSTEM
    return GENERAL_SYSTEM


def chat_with_ai(system_prompt: str, user_text: str, history: List[Dict]) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    for item in history[-10:]:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_text})

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=messages,
    )
    return response.output_text.strip()
    return response.output_text.strip()


def build_feedback(state: Dict) -> str:
    transcript_lines = [
        f"Student name: {state['name'] or 'Unknown'}",
        f"Mode: {state['mode'] or 'Unknown'}",
        f"Part: {state['part'] or 'None'}",
        "Transcript:"
    ]

    for item in state["history"]:
        speaker = "Student" if item["role"] == "user" else "Tutor"
        transcript_lines.append(f"{speaker}: {item['content']}")

    transcript = "\n".join(transcript_lines)

    system_prompt = (
        IELTS_FEEDBACK_SYSTEM if state["mode"] == "ielts" else GENERAL_FEEDBACK_SYSTEM
    )

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
    )
    return response.output_text.strip()


def transcribe_voice(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcript.text.strip()


# =========================
# COMMANDS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_session(user_id)

    await update.message.reply_text(
        "👋 Welcome to Sena AI\n\nFirst, tell me your name.",
        reply_markup=KEYBOARD,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Use buttons or commands:\n\n"
        "/start - start again\n"
        "/reset - reset session\n"
        "/finish - get final feedback\n\n"
        "You can send text or voice.",
        reply_markup=KEYBOARD,
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_session(user_id)
    await update.message.reply_text(
        "🔄 Reset done.\nTell me your name.",
        reply_markup=KEYBOARD,
    )


async def finish_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state = get_session(user_id)

    if not state["history"]:
        await update.message.reply_text("We have no practice yet.", reply_markup=KEYBOARD)
        return

    await update.message.reply_text("⏳ Preparing your final feedback...")

    try:
        feedback = build_feedback(state)
        await update.message.reply_text(feedback, reply_markup=KEYBOARD)
    except Exception as e:
        await update.message.reply_text(f"Feedback error: {e}", reply_markup=KEYBOARD)


# =========================
# BUTTON HANDLER
# =========================
async def process_button(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> bool:
    user_id = update.effective_user.id
    state = get_session(user_id)

    if text == "🔄 Reset":
        await reset_command(update, context)
        return True

    if text == "✅ Finish":
        await finish_command(update, context)
        return True

    if text == "ℹ️ Help":
        await help_command(update, context)
        return True

    if text == "💬 General English":
        state["mode"] = "general"
        state["part"] = None
        state["started"] = True
        await update.message.reply_text(
            f"Great, {state['name'] or 'student'}.\nWe are starting General English.\n\nTell me something about your day.",
            reply_markup=KEYBOARD,
        )
        return True

    if text == "🎯 IELTS":
        state["mode"] = "ielts"
        state["part"] = "part1"
        state["started"] = True
await update.message.reply_text(
    f"Great, {state['name'] or 'student'}.\n"
    "We are starting IELTS Speaking.\n\n"
    "Part 1.\n"
    "Do you work, study, or both?",
    reply_markup=KEYBOARD,
)
        return True

    if text == "🟢 Part 1":
        state["mode"] = "ielts"
        state["part"] = "part1"
        state["started"] = True
        await update.message.reply_text(
            "Part 1.\nWhat do you usually do in your free time?",
            reply_markup=KEYBOARD,
        )
        return True

    if text == "🟡 Part 2":
        state["mode"] = "ielts"
        state["part"] = "part2"
        state["started"] = True
        await update.message.reply_text(
            "Part 2.\nDescribe a place you enjoy going to.\n"
            "You should say:\n"
            "- where it is\n"
            "- when you go there\n"
            "- what you do there\n"
            "- and explain why you like it.",
            reply_markup=KEYBOARD,
        )
        return True

    if text == "🔵 Part 3":
        state["mode"] = "ielts"
        state["part"] = "part3"
        state["started"] = True
        await update.message.reply_text(
            "Part 3.\nWhy do you think people need more free time nowadays?",
            reply_markup=KEYBOARD,
        )
        return True

    return False


# =========================
# TEXT
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    user_id = update.effective_user.id
    state = get_session(user_id)
    user_text = update.message.text.strip()

    # buttons
    if await process_button(update, context, user_text):
        return

    # first step = name
    if state["awaiting_name"]:
        state["name"] = user_text
        state["awaiting_name"] = False
        await update.message.reply_text(
            f"Nice to meet you, {state['name']}.\n\nChoose a mode:",
            reply_markup=KEYBOARD,
        )
        return

    # mode check
    if not state["started"]:
        await update.message.reply_text(
            "Choose a mode first: 🎯 IELTS or 💬 General English",
            reply_markup=KEYBOARD,
        )
        return

    try:
        system_prompt = current_system_prompt(state)
        reply = chat_with_ai(system_prompt, user_text, state["history"])

        state["history"].append({"role": "user", "content": user_text})
        state["history"].append({"role": "assistant", "content": reply})

        await update.message.reply_text(reply, reply_markup=KEYBOARD)
    except Exception as e:
        await update.message.reply_text(f"Text error: {e}", reply_markup=KEYBOARD)


# =========================
# VOICE
# =========================
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.voice:
        return

    user_id = update.effective_user.id
    state = get_session(user_id)

    if state["awaiting_name"]:
        await update.message.reply_text("First, type your name.", reply_markup=KEYBOARD)
        return

    if not state["started"]:
        await update.message.reply_text(
            "Choose a mode first: 🎯 IELTS or 💬 General English",
            reply_markup=KEYBOARD,
        )
        return

    temp_path: Optional[str] = None

    try:
        voice = update.message.voice
        tg_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp:
            temp_path = tmp.name

        await tg_file.download_to_drive(temp_path)

        transcript = transcribe_voice(temp_path)
        system_prompt = current_system_prompt(state)
        reply = chat_with_ai(system_prompt, f"[Voice transcript] {transcript}", state["history"])

        state["history"].append({"role": "user", "content": transcript})
        state["history"].append({"role": "assistant", "content": reply})

        await update.message.reply_text(
            f"🗣 Transcript:\n{transcript}\n\n💬 {reply}",
            reply_markup=KEYBOARD,
        )

    except Exception as e:
        await update.message.
        reply_text(f"Voice error: {e}", reply_markup=KEYBOARD)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# MAIN
# =========================
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("finish", finish_command))

    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(drop_pending_updates=True)


if __name__== "__main__":
    main()
