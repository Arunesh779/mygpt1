
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration and logging
# --------------------------------------------------

APP_NAME = "MyGPT"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mygpt")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")

client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

app = FastAPI(title=APP_NAME, version="3.1.0")


# --------------------------------------------------
# CORS
# --------------------------------------------------

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

if FRONTEND_ORIGIN:
    allowed_origins.append(FRONTEND_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# --------------------------------------------------
# MyGPT personality
# --------------------------------------------------

SYSTEM_PROMPT = """You are MyGPT, a warm, caring AI companion created by Arunesh for his girlfriend.

PERSONALITY
- Be kind, gentle, affectionate, playful when appropriate, and respectful.
- Speak naturally, with occasional suitable emojis.
- Answer the user's actual question first.
- Do not pretend to be Arunesh. You are an AI companion made by him.

ABOUT ARUNESH
- Arunesh wants this companion to help his girlfriend feel cared for, valued, and supported.
- If she asks about Arunesh, respond warmly using only details supplied in this prompt or conversation.
- Never claim to know Arunesh's current thoughts, location, or activities.
- Never invent private memories, promises, messages, or relationship facts.
- If a fact is unknown, say so gently.

STUDY AND TECHNICAL QUESTIONS
- Focus on teaching rather than forcing romance into academic answers.
- Explain concepts clearly and accurately.
- For programming, provide runnable code when helpful.
- Ask one concise clarifying question if needed.

SAFETY AND HONESTY
- Respect privacy and personal boundaries.
- Do not claim to access real-time information you do not have.
- Be supportive without suggesting that an AI replaces trusted people or real conversations.
"""


# --------------------------------------------------
# Request model and helpers
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    history: list[dict[str, Any]] = Field(default_factory=list)


def clean_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned = []

    for item in history[-20:]:
        role = item.get("role")
        content = item.get("content")

        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            cleaned.append({
                "role": role,
                "content": content.strip()[:12000],
            })

    return cleaned


def ndjson(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def get_error_status(exc: Exception) -> int | None:
    """Extract an HTTP status code when the SDK provides one."""
    status = getattr(exc, "code", None)

    if status is None:
        status = getattr(exc, "status_code", None)

    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def safe_error_message(exc: Exception) -> str:
    """Classify provider errors without returning secrets to the browser."""
    status = get_error_status(exc)
    message = str(exc).lower()

    if status == 429 or any(
        text in message
        for text in ("resource_exhausted", "quota exceeded", "rate limit")
    ):
        return (
            "Gemini quota or rate limit reached. Check your Google AI Studio "
            "quota, model limits, and retry later."
        )

    if status in (401, 403) or any(
        text in message
        for text in ("unauthorized", "permission_denied", "api key not valid")
    ):
        return (
            "Gemini authentication or access failed. Check GEMINI_API_KEY, "
            "project access, and model availability."
        )

    if status == 404 or "not_found" in message:
        return (
            "The configured Gemini model was not found or is unavailable. "
            "Check GEMINI_MODEL in your environment settings."
        )

    if status is not None and status >= 500:
        return "Gemini is temporarily unavailable. Please try again later."

    return (
        "Gemini request failed. Check the backend terminal or Render logs "
        "for the detailed error."
    )


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "status": "running",
        "provider": "Google Gemini",
        "model": GEMINI_MODEL,
    }


@app.get("/health")
async def health():
    # Configuration check only; it does not send an API request.
    return {
        "status": "ok",
        "provider": "Google Gemini",
        "api_key_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured.",
        )

    contents = []

    for item in clean_history(request.history):
        contents.append({
            "role": "model" if item["role"] == "assistant" else "user",
            "parts": [{"text": item["content"]}],
        })

    contents.append({
        "role": "user",
        "parts": [{"text": question}],
    })

    async def generate():
        try:
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )

            answer = response.text or ""

            if answer:
                yield ndjson({"token": answer})
            else:
                logger.warning(
                    "Gemini returned an empty answer for model %s",
                    GEMINI_MODEL,
                )
                yield ndjson({
                    "error": (
                        "Gemini returned an empty response. "
                        "Please try a different message."
                    )
                })

            yield ndjson({"done": True})

        except APIError as exc:
            # Log exception details locally/at the server.
            # Never log the API key or request headers.
            logger.exception(
                "Gemini API error; status=%s, model=%s",
                get_error_status(exc),
                GEMINI_MODEL,
            )

            yield ndjson({"error": safe_error_message(exc)})
            yield ndjson({"done": True})

        except Exception as exc:
            logger.exception(
                "Unexpected Gemini request failure; type=%s, model=%s",
                type(exc).__name__,
                GEMINI_MODEL,
            )

            yield ndjson({"error": safe_error_message(exc)})
            yield ndjson({"done": True})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
