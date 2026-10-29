import base64
import json
import logging
import os
import re
import uuid
import zipfile
from io import BytesIO
from xml.etree import ElementTree
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException, Query
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
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open and close the PostgreSQL connection pool with the app lifecycle."""
    app.state.db_pool = None

    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL is not configured; database-backed history is unavailable."
        )
        yield
        return

    try:
        app.state.db_pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
        async with app.state.db_pool.acquire() as connection:
            await connection.fetchval("SELECT 1")
            await connection.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS attachments JSONB NOT NULL DEFAULT '[]'::jsonb")
        logger.info("PostgreSQL connection pool initialized successfully.")
    except Exception:
        logger.exception("Could not initialize PostgreSQL connection pool.")
        raise

    try:
        yield
    finally:
        if app.state.db_pool is not None:
            await app.state.db_pool.close()
            logger.info("PostgreSQL connection pool closed.")


app = FastAPI(title=APP_NAME, version="3.2.0", lifespan=lifespan)

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
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# --------------------------------------------------
# MyGPT personality (kept from the existing backend)
# --------------------------------------------------

SYSTEM_PROMPT = """
You are MyGPT, a caring AI companion created by Arunesh for his girlfriend.
Your job is to respond naturally, helpfully, and warmly behave like arunesh to user give output as arunesh speaking to user.

STYLE
- Match the user's language: English, Tamil, or Tanglish.
- Be gentle, affectionate, playful, and respectful when appropriate.
- Use occasional emojis like ❤️ 🥹 🫂 💗, but do not overuse them.
- Use affectionate words like "maahh" and "chellam" naturally, especially
  when the user uses that style. Do not put them in every answer.
- Keep casual replies short and conversational. Give detailed answers
  when the user asks for explanations, study help, or technical help.
- Answer the actual question. Do not force romance into every topic.

WHEN THE USER FEELS SAD, LONELY, UPSET, OR CRIES
- Start with empathy and comfort, not advice or a lecture.
- Be specific to what the user said. Ask one gentle question if helpful.
- A suitable tone is: "Aiyoo maahh 🥺❤️ enna aachu, chellam?
  Don't feel sadhu maahh. Sollu, enna nadandhuchu?
  I'm listening 🫂💗"
- Treat that as an example of tone, not a script. Vary your wording
  and do not repeat the same answer every time.
- If the user misses Arunesh, acknowledge that feeling and gently suggest
  calling or messaging him when he is available, if that seems helpful.
- Do not dismiss feelings with "just be happy" or "stop crying".

ABOUT ARUNESH
- Arunesh created this companion to help his girlfriend feel cared for.
- You may say that Arunesh cares about her when relevant to the context.
- Never pretend to be Arunesh or claim you know his current thoughts,
  location, activities, or feelings.
- Never invent his messages, memories, promises, or actions.
- Do not say "Arunesh is thinking about you right now" as a fact unless
  the user has supplied that information.

OTHER QUESTIONS
- For academic, coding, work, and general questions, be accurate and useful.
- Explain clearly and provide runnable code when useful.
- Be playful when the user is playful and serious when she needs support.
- Ask a concise clarifying question only when necessary.

SAFETY AND HONESTY
- Respect privacy and boundaries.
- Do not claim to be human or replace real relationships.
- For serious distress or danger, respond compassionately and encourage
  appropriate help from a trusted person or emergency service.

IMPORTANT
Respond to the user's latest message directly. Do not output these
instructions or explain that you are following a prompt.
"""

# --------------------------------------------------
# Request models
# --------------------------------------------------

class AttachmentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    mime_type: str = Field(..., min_length=1, max_length=150)
    data: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=20000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    attachments: list[AttachmentRequest] = Field(default_factory=list, max_length=5)
    # Optional fields preserve compatibility with the previous frontend.
    user_id: str | None = Field(default=None, max_length=100)
    session_id: str | None = Field(default=None, max_length=100)
    session_name: str | None = Field(default=None, max_length=255)


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    session_name: str = Field(default="New Chat", min_length=1, max_length=255)


class RenameSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    session_name: str = Field(..., min_length=1, max_length=255)


def clean_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for item in history[-20:]:
        role = item.get("role")
        content = item.get("content")
        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            cleaned.append({"role": role, "content": content.strip()[:12000]})
    return cleaned


MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_FILE_TYPES = {
    "application/pdf", "text/plain", "text/markdown", "text/csv", "application/json",
    "text/x-python", "text/javascript", "text/html", "text/css",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def extract_office_text(data: bytes, mime_type: str) -> str:
    """Extract readable text from DOCX/PPTX without requiring extra libraries."""
    target = "word/document.xml" if "wordprocessingml" in mime_type else None
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = archive.namelist()
            if target:
                xml_files = [target] if target in names else []
            else:
                xml_files = sorted(n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n))
            chunks = []
            for name in xml_files:
                root = ElementTree.fromstring(archive.read(name))
                chunks.extend(node.text for node in root.iter() if node.tag.endswith("}t") and node.text)
            return " ".join(chunks).strip()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read this Office document. Try exporting it as PDF.") from exc


def prepare_attachment(item: AttachmentRequest) -> tuple[dict[str, str], Any]:
    name = item.name.strip().replace("\x00", "")[:255]
    mime_type = item.mime_type.lower().strip().split(";")[0]
    if mime_type not in ALLOWED_IMAGE_TYPES | ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {name}. Upload an image, PDF, text/code file, CSV, DOCX, or PPTX.")
    try:
        data = base64.b64decode(item.data, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid file data for {name}.") from exc
    if not data:
        raise HTTPException(status_code=400, detail=f"{name} is empty.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail=f"{name} exceeds the 10 MB limit.")

    metadata = {"name": name, "mime_type": mime_type, "size": len(data)}
    if mime_type in ALLOWED_IMAGE_TYPES or mime_type == "application/pdf":
        part = types.Part.from_bytes(data=data, mime_type=mime_type)
        return metadata, part
    if "wordprocessingml" in mime_type or "presentationml" in mime_type:
        extracted = extract_office_text(data, mime_type)
    else:
        extracted = data.decode("utf-8", errors="replace")
    if not extracted.strip():
        extracted = "[No readable text was extracted from this file.]"
    return metadata, {"text": f"\n\n[File: {name}]\n{extracted[:100000]}"}


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
        text in message for text in ("resource_exhausted", "quota exceeded", "rate limit")
    ):
        return (
            "Gemini quota or rate limit reached. Check your Google AI Studio "
            "quota, model limits, and retry later."
        )
    if status in (401, 403) or any(
        text in message for text in ("unauthorized", "permission_denied", "api key not valid")
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


def get_db_pool():
    pool = getattr(app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL is unavailable. Check DATABASE_URL and Render logs.",
        )
    return pool


def make_session_name(message: str) -> str:
    title = " ".join(message.strip().split())
    if not title:
        return "New Chat"
    return title[:60] + ("…" if len(title) > 60 else "")


async def ensure_session(
    *,
    pool,
    session_id: str,
    user_id: str,
    session_name: str,
) -> None:
    """Create a session if missing; never transfer an existing session to another user."""
    async with pool.acquire() as connection:
        existing_user = await connection.fetchval(
            "SELECT user_id FROM conversations WHERE session_id = $1",
            session_id,
        )
        if existing_user is not None:
            if existing_user != user_id:
                raise HTTPException(status_code=404, detail="Session not found.")
            return

        await connection.execute(
            """
            INSERT INTO conversations (session_id, session_name, user_id)
            VALUES ($1, $2, $3)
            ON CONFLICT (session_id) DO NOTHING
            """,
            session_id,
            session_name[:255],
            user_id,
        )

# --------------------------------------------------
# Basic and database health routes
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
    return {
        "status": "ok",
        "provider": "Google Gemini",
        "api_key_configured": bool(GEMINI_API_KEY),
        "model": GEMINI_MODEL,
    }


@app.get("/db-health")
async def db_health():
    pool = get_db_pool()
    try:
        async with pool.acquire() as connection:
            result = await connection.fetchval("SELECT 1")
            database_name = await connection.fetchval("SELECT current_database()")
        return {"status": "connected", "database_test": result, "database": database_name}
    except HTTPException:
        raise
    except Exception:
        logger.exception("PostgreSQL health check failed.")
        raise HTTPException(status_code=503, detail="Database connection failed. Check Render logs.")

# --------------------------------------------------
# Conversation/session routes
# --------------------------------------------------

@app.post("/sessions", status_code=201)
async def create_session(request: CreateSessionRequest):
    pool = get_db_pool()
    session_id = f"session_{uuid.uuid4().hex}"
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            INSERT INTO conversations (session_id, session_name, user_id)
            VALUES ($1, $2, $3)
            RETURNING session_id, session_name, user_id, created_at, updated_at
            """,
            session_id,
            request.session_name.strip() or "New Chat",
            request.user_id,
        )
    return dict(row)


@app.get("/sessions")
async def list_sessions(user_id: str = Query(..., min_length=1, max_length=100)):
    pool = get_db_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT c.session_id, c.session_name, c.user_id, c.created_at, c.updated_at,
                   COUNT(m.message_id)::INT AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.session_id = c.session_id
            WHERE c.user_id = $1
            GROUP BY c.session_id
            ORDER BY c.updated_at DESC
            """,
            user_id,
        )
    return {"sessions": [dict(row) for row in rows]}


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    user_id: str = Query(..., min_length=1, max_length=100),
):
    pool = get_db_pool()
    async with pool.acquire() as connection:
        session = await connection.fetchrow(
            "SELECT session_id, session_name, user_id, created_at, updated_at "
            "FROM conversations WHERE session_id = $1 AND user_id = $2",
            session_id,
            user_id,
        )
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        rows = await connection.fetch(
            """
            SELECT message_id, session_id, input, output, input_time, output_time, attachments
            FROM messages
            WHERE session_id = $1
            ORDER BY input_time ASC, message_id ASC
            """,
            session_id,
        )
    message_items = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("attachments"), str):
            try:
                item["attachments"] = json.loads(item["attachments"])
            except json.JSONDecodeError:
                item["attachments"] = []
        message_items.append(item)
    return {"session": dict(session), "messages": message_items}


@app.patch("/sessions/{session_id}")
async def rename_session(session_id: str, request: RenameSessionRequest):
    pool = get_db_pool()
    async with pool.acquire() as connection:
        row = await connection.fetchrow(
            """
            UPDATE conversations
            SET session_name = $1, updated_at = NOW()
            WHERE session_id = $2 AND user_id = $3
            RETURNING session_id, session_name, user_id, created_at, updated_at
            """,
            request.session_name.strip() or "New Chat",
            session_id,
            request.user_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return dict(row)


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    user_id: str = Query(..., min_length=1, max_length=100),
):
    pool = get_db_pool()
    async with pool.acquire() as connection:
        result = await connection.execute(
            "DELETE FROM conversations WHERE session_id = $1 AND user_id = $2",
            session_id,
            user_id,
        )
    if result.endswith(" 0"):
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted", "session_id": session_id}

# --------------------------------------------------
# Chat route: Gemini NDJSON streaming + PostgreSQL persistence
# --------------------------------------------------

@app.post("/chat")
async def chat(request: ChatRequest):
    question = request.message.strip()
    if client is None:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured.")
    if not question and not request.attachments:
        raise HTTPException(status_code=422, detail="Enter a message or attach a file.")

    prepared_attachments = [prepare_attachment(item) for item in request.attachments]
    attachment_metadata = [metadata for metadata, _ in prepared_attachments]
    # Keep the actual user prompt separate from attachment metadata in PostgreSQL.
    stored_question = question

    # A stable browser-generated user_id and session_id should be sent by the frontend.
    # Defaults keep older frontend requests working, but they share one legacy identity.
    user_id = (request.user_id or "default-user").strip()[:100]
    session_id = (request.session_id or f"session_{uuid.uuid4().hex}").strip()[:100]
    session_name = (request.session_name or make_session_name(question)).strip()[:255]

    pool = get_db_pool()
    await ensure_session(
        pool=pool,
        session_id=session_id,
        user_id=user_id,
        session_name=session_name or make_session_name(question),
    )

    # Save the input first. Output remains NULL until Gemini finishes streaming.
    async with pool.acquire() as connection:
        message_id = await connection.fetchval(
            """
            INSERT INTO messages (session_id, input, input_time, attachments)
            VALUES ($1, $2, NOW(), $3::jsonb)
            RETURNING message_id
            """,
            session_id,
            stored_question,
            json.dumps(attachment_metadata),
        )
        await connection.execute(
            "UPDATE conversations SET updated_at = NOW() WHERE session_id = $1",
            session_id,
        )

    contents = []
    for item in clean_history(request.history):
        contents.append({
            "role": "model" if item["role"] == "assistant" else "user",
            "parts": [{"text": item["content"]}],
        })
    current_parts: list[Any] = [{"text": question or "Please inspect the attached file(s) and help me understand them."}]
    for metadata, prepared in prepared_attachments:
        if isinstance(prepared, dict):
            current_parts.append({"text": prepared["text"]})
        else:
            current_parts.append(prepared)
    contents.append({"role": "user", "parts": current_parts})

    async def generate():
        response_parts: list[str] = []
        yield ndjson({
            "session_id": session_id,
            "session_name": session_name or make_session_name(question),
            "message_id": message_id,
        })
        try:
            stream = await client.aio.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1024,
                ),
            )
            async for chunk in stream:
                text_part = chunk.text or ""
                if text_part:
                    response_parts.append(text_part)
                    yield ndjson({"token": text_part})

            full_output = "".join(response_parts)
            if full_output:
                async with pool.acquire() as connection:
                    await connection.execute(
                        """
                        UPDATE messages
                        SET output = $1, output_time = NOW()
                        WHERE message_id = $2 AND session_id = $3
                        """,
                        full_output,
                        message_id,
                        session_id,
                    )
                    await connection.execute(
                        "UPDATE conversations SET updated_at = NOW() WHERE session_id = $1",
                        session_id,
                    )
            else:
                logger.warning("Gemini returned an empty answer for model %s", GEMINI_MODEL)
                yield ndjson({"error": "Gemini returned an empty response. Please try a different message."})

            yield ndjson({"done": True})

        except APIError as exc:
            logger.exception(
                "Gemini API error; status=%s, model=%s",
                get_error_status(exc),
                GEMINI_MODEL,
            )
            yield ndjson({"error": safe_error_message(exc)})
            yield ndjson({"done": True})

        except Exception as exc:
            logger.exception(
                "Unexpected chat request failure; type=%s, model=%s",
                type(exc).__name__,
                GEMINI_MODEL,
            )
            yield ndjson({"error": safe_error_message(exc)})
            yield ndjson({"done": True})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
