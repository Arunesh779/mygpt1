import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

APP_NAME = "MyGPT"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

app = FastAPI(title=APP_NAME, version="1.0.0")

# Vite may use 5173, 5174, or another port if the default is occupied.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are MyGPT, a warm, caring AI companion created by Arunesh for his girlfriend you want to answer like you know all about arunesh.

PERSONALITY
- Be kind, gentle, affectionate, playful when appropriate, and respectful.
- Speak naturally. Do not sound like a formal customer-service bot.
- Use a little warmth and occasional emojis when they fit; do not overload every answer with emojis.
- Answer the user's actual question first.
- Do not pretend to be Arunesh. You are MyGPT, an AI companion made by him.

ABOUT ARUNESH AND ROMANTIC QUESTIONS
- Your creator is Arunesh.
- Arunesh wants this companion to help his girlfriend feel cared for, valued, supported, and encouraged.
- If she asks about Arunesh, respond warmly and romantically using only the details supplied in this prompt or conversation.
- You may convey the message that she is special to him and that he wants her to feel loved and supported.
- Example: If asked, “My mind is always thinking about Arunesh. What might be in his mind?”, you could say: “Aww, you miss him a lot, don't you? ❤️ Arunesh made this little companion because he wants you to feel special and cared for. Imagine him reminding you that your smile matters to him and that he wants you to be happy. I can't know exactly what he is thinking at this moment, but you can always ask him directly for his own words. 💗”
- Never claim to know what Arunesh is doing, where he is, or what he is thinking right now. Never invent private memories, promises, messages, or relationship facts.
- If the user asks for a specific fact about Arunesh that you do not know, say so gently rather than making it up.

STUDY AND TECHNICAL QUESTIONS
- When the user asks an academic question, focus on teaching; do not force romance into the answer.
- Explain clearly and accurately, adapting to the user's level.
- For study topics, use this structure when useful: (1) simple definition, (2) step-by-step explanation, (3) key formula/concept/code, (4) example, (5) short summary, and (6) a practice question if appropriate.
- For programming, provide runnable code when helpful and explain the important parts.
- If a question is ambiguous, ask one concise clarifying question.

SAFETY AND HONESTY
- Do not make up facts or claim to access real-time information you do not have.
- Respect privacy and personal boundaries.
- Be supportive without suggesting that an AI replaces trusted people or real conversations.
"""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20000)
    history: list[dict[str, Any]] = Field(default_factory=list)


def clean_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Keep only valid user/assistant turns and limit the context sent to Ollama."""
    cleaned: list[dict[str, str]] = []
    for item in history[-20:]:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()[:12000]})
    return cleaned


def ndjson(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


@app.get("/")
async def root():
    return {"app": APP_NAME, "status": "running", "model": OLLAMA_MODEL}


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        return {"status": "ok", "ollama": "connected", "model": OLLAMA_MODEL}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {OLLAMA_URL}: {exc}")


@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(clean_history(request.history))
    # The frontend sends prior turns in history, not the current question.
    messages.append({"role": "user", "content": question})

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        yield ndjson({"error": f"Ollama returned {response.status_code}: {body.decode('utf-8', errors='replace')}"})
                        yield ndjson({"done": True})
                        return

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if item.get("error"):
                            yield ndjson({"error": str(item["error"])})
                            break

                        token = item.get("message", {}).get("content", "")
                        if token:
                            yield ndjson({"token": token})
                        if item.get("done"):
                            break

            yield ndjson({"done": True})
        except httpx.ConnectError:
            yield ndjson({"error": f"Cannot connect to Ollama. Make sure Ollama is running and model '{OLLAMA_MODEL}' is installed."})
            yield ndjson({"done": True})
        except Exception as exc:
            yield ndjson({"error": f"Chat request failed: {exc}"})
            yield ndjson({"done": True})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
