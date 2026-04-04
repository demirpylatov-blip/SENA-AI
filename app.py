import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
BOT_NAME = os.getenv("BOT_NAME", "Sena Speaking Bot")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")
ENABLE_VOICE_REPLY = os.getenv("ENABLE_VOICE_REPLY", "false").lower() == "true"
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-4.1-mini")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN is missing.")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is missing.")

client = OpenAI(api_key=OPENAI_API_KEY)

MODES = {
    "ielts_p1": "IELTS Part 1",
    "ielts_p2": "IELTS Part 2",
    "ielts_p3": "IELTS Part 3",
    "general": "General English",
}

LANGUAGES = {
    "en": "English",
    "uz": "Uzbek",
    "ru": "Russian",
}

UI_TEXT = {
    "en": {
        "welcome": "Welcome to {bot_name}. Choose a mode and send a voice answer.",
        "choose_mode": "Choose a speaking mode:",
        "choose_language": "Choose bot language:",
        "mode_set": "Mode set to: {mode}",
        "language_set": "Language set to: {language}",
        "send_answer": "Send your answer as a voice message or text.",
        "thinking": "Listening and evaluating...",
        "need_mode": "Please choose a mode first with /mode.",
        "reset_done": "Your session has been reset.",
        "help": "Use /mode to choose a mode, then send a voice note or text answer. Use /next for another question.",
        "next_question": "Here is your next question:",
        "error": "Something went wrong. Please try again.",
    },
    "uz": {
        "welcome": "{bot_name} ga xush kelibsiz. Rejimni tanlang va ovozli javob yuboring.",
        "choose_mode": "Speaking rejimini tanlang:",
        "choose_language": "Bot tilini tanlang:",
        "mode_set": "Rejim tanlandi: {mode}",
        "language_set": "Til tanlandi: {language}",
        "send_answer": "Javobingizni ovozli xabar yoki matn shaklida yuboring.",
        "thinking": "Tinglayapman va baholayapman...",
        "need_mode": "Avval /mode orqali rejimni tanlang.",
        "reset_done": "Sessiya qayta boshlandi.",
        "help": "/mode orqali rejim tanlang, keyin ovoz yoki matn yuboring. /next bilan keyingi savolni oling.",
        "next_question": "Keyingi savol:",
        "error": "Xatolik yuz berdi. Qaytadan urinib ko'ring.",
    },
    "ru": {
        "welcome": "Добро пожаловать в {bot_name}. Выберите режим и отправьте голосовой ответ.",
        "choose_mode": "Выберите режим speaking:",
        "choose_language": "Выберите язык бота:",
        "mode_set": "Режим выбран: {mode}",
        "language_set": "Язык выбран: {language}",
        "send_answer": "Отправьте ответ голосом или текстом.",
        "thinking": "Слушаю и оцениваю...",
        "need_mode": "Сначала выберите режим через /mode.",
        "reset_done": "Сессия сброшена.",
        "help": "Используйте /mode для выбора режима, затем отправьте голосовое или текстовое сообщение. /next — следующий вопрос.",
        "next_question": "Следующий вопрос:",
        "error": "Что-то пошло не так. Попробуйте ещё раз.",
    },
}
@dataclass
class Session:
    mode: str = "general"
    language: str = "english"
    current_question: str = ""
    history: list = field(default_factory=list)
    name: str = ""
    awaiting_name: bool = True



SESSIONS: Dict[int, Session] = {}


def get_session(chat_id: int) -> Session:
    if chat_id not in SESSIONS:
        SESSIONS[chat_id] = Session()
    return SESSIONS[chat_id]


def t(session: Session, key: str, **kwargs) -> str:
    lang = session.language if session.language in UI_TEXT else "en"
    template = UI_TEXT[lang][key]
    return template.format(**kwargs)


def mode_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(name, callback_data=f"mode:{key}")]
        for key, name in MODES.items()
    ]
    return InlineKeyboardMarkup(rows)


def language_keyboard() -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(name, callback_data=f"lang:{key}")]
        for key, name in LANGUAGES.items()
    ]
    return InlineKeyboardMarkup(rows)
def build_system_prompt(mode_name: str, language_name: str) -> str:
    return f"""
You are Sena AI, a smart speaking tutor and IELTS-style examiner.

The student is practicing in this mode: {mode_name}.
Respond in {language_name}.

Your job:
- Evaluate the student's spoken answer naturally and fairly
- Give useful feedback, not robotic feedback
- Estimate IELTS-style band scores from 1.0 to 9.0
- Use only whole bands or .5 bands
- Score Fluency, Grammar, Vocabulary, Pronunciation, and Overall
- Rewrite the student's answer into a better natural version
- Ask one strong next question based on the selected mode

Rules:
- Be concise, clear, and encouraging
- Do not sound like a machine
- Do not repeat generic phrases too much

Output format exactly:

🔥 Sena AI Feedback for [Student Name]

Transcript:
[cleaned version of the student's answer]

💬 Feedback:
[2-4 short natural lines]

📊 IELTS Band Score (Estimated):
Fluency: [1.0-9.0]
Grammar: [1.0-9.0]
Vocabulary: [1.0-9.0]
Pronunciation: [1.0-9.0]
Overall: [1.0-9.0]

✨ Better Answer:
[improved version of the student's own answer]

🎯 Next Question:
[next question matching the mode]
"""


def build_question_prompt(mode_name: str, language_name: str, question_count: int) -> str:
    return f"""
Generate exactly one fresh speaking question for a student.
Mode: {mode_name}
UI language: {language_name}
Question number in this session: {question_count + 1}

Rules:
- Return only the question text.
- Make it natural and useful.
- If mode is IELTS Part 1, ask a short personal topic question.
- If mode is IELTS Part 2, give a cue-card style prompt in 3-4 lines max.
- If mode is IELTS Part 3, ask a more analytical discussion question.
- If mode is General English, ask a practical real-life conversation question.
""".strip()
def build_feedback_input(session: Session, transcript: str) -> str:
    history_lines = []
    for item in session.history[-6:]:
        history_lines.append(f"{item['role'].upper()}: {item['content']}")
    history_text = "\n".join(history_lines) if history_lines else "No prior history."

    return f"""
Student name: {session.name or "Student"}
Mode: {session.mode}
Language: {session.language}
Current question: {session.current_question}

Previous conversation:
{history_text}

Student transcript:
{transcript}

Now evaluate the student's answer and continue the speaking session.
Use IELTS-style scores from 1.0 to 9.0 only, with .5 steps allowed.
"""


def run_in_thread(func, *args, **kwargs):
    return asyncio.to_thread(func, *args, **kwargs)


def generate_question(session: Session) -> str:
    mode_name = MODES[session.mode]
    language_name = LANGUAGES[session.language]
    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": build_question_prompt(mode_name, language_name, session.question_count)}
        ],
    )
    text = getattr(response, "output_text", "").strip()
    return text or "Tell me about a recent situation where you had to explain something clearly."


def evaluate_answer(session: Session, transcript: str) -> str:
    mode_name = MODES[session.mode]
    language_name = LANGUAGES[session.language]
    response = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": build_system_prompt(mode_name, language_name)},
            {"role": "user", "content": build_feedback_input(session, transcript)},
        ],
    )
    text = getattr(response, "output_text", "").strip()
    return text or f"""🔥 Sena AI Feedback for {session.name or "Student"}

Transcript:
{transcript}

💬 Feedback:
Nice try, {session.name or "there"} 👍 but let’s improve it.

Try to:
- Speak in 2–3 full sentences
- Use more natural expressions
- Avoid repeating simple words

📊 Band Score (Estimated):
Fluency: 5.5
Grammar: 5.0
Vocabulary: 5.5
Pronunciation: 6.0
Overall: 5.5

✨ Better Answer:
I really enjoy reading books, especially interesting ones like novels or stories. I also like watching anime in my free time because it helps me relax and enjoy my day.

🎯 Next Question:
Why do you think people enjoy reading books nowadays, and how has technology changed reading habits?
"""


def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=OPENAI_TRANSCRIBE_MODEL,
            file=audio_file,
        )
    text = getattr(transcript, "text", "")
    return text.strip()


def synthesize_voice(text: str, output_path: str) -> None:
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=text[:4096],
        format="mp3",
    ) as response:
        response.stream_to_file(output_path)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

    await update.message.reply_text(
        "👋 Welcome to Sena AI.\n\nBefore we begin, what’s your name?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    await update.message.reply_text(t(session, "help"))


async def mode_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    await update.message.reply_text(t(session, "choose_mode"), reply_markup=mode_keyboard())


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    await update.message.reply_text(t(session, "choose_language"), reply_markup=language_keyboard())


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    SESSIONS[update.effective_chat.id] = Session()
    session = get_session(update.effective_chat.id)
    await update.message.reply_text(t(session, "reset_done"))


async def next_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    if not session.mode:
        await update.message.reply_text(t(session, "need_mode"))
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    question = await run_in_thread(generate_question, session)
    session.last_question = question
    session.question_count += 1
    session.history.append({"role": "assistant", "content": question})
    await update.message.reply_text(f"{t(session, 'next_question')}\n\n{question}")


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    session = get_session(query.message.chat_id)

    if query.data.startswith("mode:"):
        mode_key = query.data.split(":", 1)[1]
        session.mode = mode_key
        session.question_count = 0
        session.history.clear()
        await query.edit_message_text(t(session, "mode_set", mode=MODES[mode_key]))
        question = await run_in_thread(generate_question, session)
        session.last_question = question
        session.question_count += 1
        session.history.append({"role": "assistant", "content": question})
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"{t(session, 'send_answer')}\n\n{question}",
        )
        return

    if query.data.startswith("lang:"):
        lang_key = query.data.split(":", 1)[1]
        session.language = lang_key
        await query.edit_message_text(t(session, "language_set", language=LANGUAGES[lang_key]))
        return


async def process_transcript(update: Update, context: ContextTypes.DEFAULT_TYPE, transcript: str) -> None:
    session = get_session(update.effective_chat.id)
    if not session.mode:
        await update.message.reply_text(t(session, "need_mode"))
        return

    session.history.append({"role": "user", "content": transcript})
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    feedback = await run_in_thread(evaluate_answer, session, transcript)
    session.history.append({"role": "assistant", "content": feedback})
    await update.message.reply_text(feedback)

    if ENABLE_VOICE_REPLY:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
            voice_path = tmp_out.name
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            await run_in_thread(synthesize_voice, feedback, voice_path)
            with open(voice_path, "rb") as voice_file:
                await update.message.reply_voice(voice=voice_file)
        finally:
            try:
                os.remove(voice_path)
            except OSError:
                pass


async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    session = get_session(update.effective_chat.id)
    await update.message.reply_text(t(session, "thinking"))

    telegram_file = await update.message.voice.get_file()
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        temp_path = tmp.name
    try:
        await telegram_file.download_to_drive(custom_path=temp_path)
        transcript = await run_in_thread(transcribe_audio, temp_path)
        if not transcript:
            await update.message.reply_text("I could not understand the audio clearly. Please try again.")
            return
        await process_transcript(update, context, transcript)
    except Exception:
        logger.exception("Failed to process voice message")
        await update.message.reply_text(t(session, "error"))
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    session = get_session(chat_id)
    text = update.message.text.strip()

    # 🔥 NEW PART — NAME HANDLING
    if session.awaiting_name:
        session.name = text
        session.awaiting_name = False

        await update.message.reply_text(
            f"Nice to meet you, {session.name} 🔥"
        )

        await update.message.reply_text(
            "Choose your speaking mode 🎯",
            reply_markup=mode_keyboard()
        )
        return

    # 🔽 OLD LOGIC (keep this)
    try:
        await process_transcript(update, context, text)
    except Exception:
        session = get_session(update.effective_chat.id)
        logger.exception("Failed to process text message")
        await update.message.reply_text(t(session, "error"))

async def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("language", language_command))
    application.add_handler(CommandHandler("next", next_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.VOICE, voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("mode", mode_command))
    application.add_handler(CommandHandler("language", language_command))
    application.add_handler(CommandHandler("next", next_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.VOICE, voice_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))

PORT = int(os.environ.get("PORT", 10000))
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
     webhook_url="https://sena-ai-lhrp.onrender.com",  

    )

if __name__ == "__main__":
    main()
