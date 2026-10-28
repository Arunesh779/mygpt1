
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, APIError, APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field

APP_NAME = "MyGPT"

# Configure these in Render Environment.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "").strip().rstrip("/")

# Create the client only when a key is configured.
client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
    if OPENAI_API_KEY
    else None
)

app = FastAPI(title=APP_NAME, version="2.0.0")

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


@app.get("/")
async def root():
    return {
        "app": APP_NAME,
        "status": "running",
        "provider": "OpenAI",
        "model": OPENAI_MODEL,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": "OpenAI",
        "api_key_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Message cannot be empty.",
        )

    if not OPENAI_API_KEY or client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured in Render.",
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(clean_history(request.history))
    messages.append({"role": "user", "content": question})

    async def generate():
        try:
            stream = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                token = chunk.choices[0].delta.content

                if token:
                    yield ndjson({"token": token})

            yield ndjson({"done": True})

        except APITimeoutError:
            yield ndjson({
                "error": "OpenAI timed out. Please try again."
            })
            yield ndjson({"done": True})

        except APIConnectionError:
            yield ndjson({
                "error": "Could not connect to the OpenAI API."
            })
            yield ndjson({"done": True})

        except APIError as exc:
            status_code = getattr(exc, "status_code", None)

            if status_code == 401:
                detail = "OpenAI rejected the API key. Check OPENAI_API_KEY in Render."
            elif status_code == 429:
                detail = (
                    "OpenAI API quota or rate limit reached. "
                    "Check API billing, credits, and usage limits."
                )
            elif status_code == 403:
                detail = (
                    "OpenAI API access was denied. Check project permissions "
                    "and model access."
                )
            else:
                detail = f"OpenAI API error (HTTP {status_code or 'unknown'})."

            yield ndjson({"error": detail})
            yield ndjson({"done": True})

        except Exception:
            # Avoid sending internal details or secrets to the browser.
            yield ndjson({
                "error": "An unexpected chat error occurred. Check Render logs."
            })
            yield ndjson({"done": True})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
