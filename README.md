# Sena Speaking Bot (Telegram)

A Telegram AI speaking bot for IELTS and General English practice.

## What it does
- lets students choose a mode: IELTS Part 1, Part 2, Part 3, or General English
- asks speaking questions
- accepts **voice messages** or **text answers**
- transcribes audio with the OpenAI speech-to-text API
- gives structured feedback for:
  - fluency
  - grammar
  - vocabulary
  - pronunciation
  - overall estimate
- asks the next question automatically
- can optionally send a voice reply

## Stack
- Telegram Bot API via `python-telegram-bot`
- OpenAI Audio API for transcription
- OpenAI Responses API for question generation + feedback
- Optional OpenAI speech API for voice replies

## 1) Create the Telegram bot
1. Open Telegram and message **@BotFather**.
2. Use `/newbot`.
3. Copy the bot token.

Telegram supports bots over its HTTP Bot API, including voice messages and file handling. Official docs: Telegram Bot API.  
https://core.telegram.org/bots/api

## 2) Set up OpenAI
Create an API key in your OpenAI account and place it in `.env`.

OpenAI recommends the **Responses API** for new text projects, and the Audio API supports speech-to-text and text-to-speech. Official docs:  
https://developers.openai.com/api/docs/guides/migrate-to-responses/  
https://developers.openai.com/api/docs/guides/speech-to-text/  
https://developers.openai.com/api/docs/guides/text-to-speech/

## 3) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in your `.env` values.

## 4) Run
```bash
python app.py
```

## 5) Commands
- `/start` — show menu and welcome
- `/mode` — choose speaking mode
- `/language` — choose bot language
- `/next` — get the next question
- `/reset` — reset session
- `/help` — instructions

## Notes
- Telegram voice messages are typically sent as voice notes; the bot downloads them before transcription.
- The transcription endpoint supports common audio formats and has a 25 MB upload limit in OpenAI docs.
- For a premium real-time version later, you could move to a Realtime voice architecture.

## Suggested next upgrades
- save student history to SQLite/Postgres
- add teacher dashboard
- add paid plans
- add Uzbek/Russian UI strings
- add band history charts
- add Telegram Mini App
