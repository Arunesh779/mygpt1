
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API = "https://mygpt-tcnq.onrender.com";

const CHATS_KEY = "mygpt_saved_chats_v1";
const CURRENT_KEY = "mygpt_current_chat_v1";
const USER_KEY = "mygpt_browser_user_v1";

function getOrCreateUserId() {
  let userId = localStorage.getItem(USER_KEY);
  if (!userId) {
    userId = `user-${makeId()}`;
    localStorage.setItem(USER_KEY, userId);
  }
  return userId;
}

function toTimestamp(value) {
  if (!value) return Date.now();
  const parsed = typeof value === "number" ? value : Date.parse(value);
  return Number.isFinite(parsed) ? parsed : Date.now();
}

function formatDateTime(value) {
  if (!value) return "";
  const date = new Date(toTimestamp(value));
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

const makeId = () =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

const makeChat = () => ({
  id: makeId(),
  title: "New conversation",
  updatedAt: Date.now(),
  messages: [],
});

const titleFrom = (text) => {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 36
    ? `${clean.slice(0, 36)}…`
    : clean || "New conversation";
};

function readChats() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]");

    return Array.isArray(parsed)
      ? parsed.filter(
          (chat) => chat && chat.id && Array.isArray(chat.messages)
        )
      : [];
  } catch {
    return [];
  }
}

function TypingMarkdown({ text, speed = 18 }) {
  const [visibleText, setVisibleText] = useState("");

  useEffect(() => {
    // If the target text changed, reveal it gradually instead of all at once.
    if (visibleText.length >= text.length) return;

    const timer = setTimeout(() => {
      // Reveal a few characters per tick for a natural typing pace.
      const remaining = text.length - visibleText.length;
      const step = remaining > 500 ? 4 : remaining > 180 ? 3 : 2;
      setVisibleText(text.slice(0, visibleText.length + step));
    }, speed);

    return () => clearTimeout(timer);
  }, [text, visibleText, speed]);

  // If a conversation is reopened and its text is shorter, sync to that text.
  useEffect(() => {
    if (visibleText.length > text.length) {
      setVisibleText(text);
    }
  }, [text, visibleText]);

  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]}>
      {visibleText}
    </ReactMarkdown>
  );
}

export default function App() {
  const [chats, setChats] = useState(() => readChats());
  const [userId] = useState(() => getOrCreateUserId());
  const [currentId, setCurrentId] = useState(
    () => localStorage.getItem(CURRENT_KEY) || ""
  );
  const [input, setInput] = useState("");
  const [attachments, setAttachments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [error, setError] = useState("");

  const messagesContainerRef = useRef(null);
  const shouldAutoScrollRef = useRef(true);
  const abortRef = useRef(null);
  const fileInputRef = useRef(null);

  const sortedChats = useMemo(
    () =>
      [...chats].sort(
        (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)
      ),
    [chats]
  );

  const currentChat = chats.find((chat) => chat.id === currentId) || null;
  const messages = currentChat?.messages || [];

  // PostgreSQL is the shared source of truth for this browser user's history.
  const loadDatabaseChats = useCallback(async (preferredId = "") => {
    const sessionsResponse = await fetch(
      `${API}/sessions?user_id=${encodeURIComponent(userId)}`
    );
    if (!sessionsResponse.ok) {
      throw new Error(`Could not load conversation history (${sessionsResponse.status}).`);
    }

    const sessionsData = await sessionsResponse.json();
    const sessions = Array.isArray(sessionsData.sessions)
      ? sessionsData.sessions
      : [];

    const databaseChats = await Promise.all(
      sessions.map(async (session) => {
        const messagesResponse = await fetch(
          `${API}/sessions/${encodeURIComponent(session.session_id)}/messages?user_id=${encodeURIComponent(userId)}`
        );
        if (!messagesResponse.ok) {
          throw new Error(`Could not load messages for ${session.session_name || "a conversation"}.`);
        }

        const messagesData = await messagesResponse.json();
        const restoredMessages = [];

        for (const row of messagesData.messages || []) {
          restoredMessages.push({
            id: `${row.message_id}-user`,
            role: "user",
            content: row.input || "",
            attachments: Array.isArray(row.attachments) ? row.attachments : [],
            timestamp: toTimestamp(row.input_time),
          });

          if (row.output !== null && row.output !== undefined && row.output !== "") {
            restoredMessages.push({
              id: `${row.message_id}-assistant`,
              role: "assistant",
              content: row.output,
              timestamp: toTimestamp(row.output_time || row.input_time),
            });
          }
        }

        return {
          id: session.session_id,
          title: session.session_name || "New conversation",
          updatedAt: toTimestamp(session.updated_at || session.created_at),
          createdAt: toTimestamp(session.created_at),
          messages: restoredMessages,
        };
      })
    );

    // Keep local drafts as a fallback, but prefer database records for matching IDs.
    setChats((previous) => {
      const databaseIds = new Set(databaseChats.map((chat) => chat.id));
      const localOnly = previous.filter((chat) => !databaseIds.has(chat.id));
      return [...databaseChats, ...localOnly].sort(
        (a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)
      );
    });

    const savedCurrentId = localStorage.getItem(CURRENT_KEY) || "";
    const nextCurrentId =
      databaseChats.find((chat) => chat.id === preferredId)?.id ||
      databaseChats.find((chat) => chat.id === savedCurrentId)?.id ||
      databaseChats[0]?.id;

    if (nextCurrentId) setCurrentId(nextCurrentId);
    return databaseChats;
  }, [userId]);

  useEffect(() => {
    loadDatabaseChats().catch((err) => {
      console.error("Could not restore MyGPT history from PostgreSQL:", err);
    });
  }, [loadDatabaseChats]);

  // Save chats locally.
  useEffect(() => {
    // Keep attachment bytes out of localStorage to avoid exhausting browser storage.
    const cacheSafeChats = chats.map((chat) => ({
      ...chat,
      messages: (chat.messages || []).map((message) => ({
        ...message,
        attachments: (message.attachments || []).map(({ id, name, mime_type, size }) => ({ id, name, mime_type, size })),
      })),
    }));
    try {
      localStorage.setItem(CHATS_KEY, JSON.stringify(cacheSafeChats));
    } catch (err) {
      console.warn("MyGPT could not update the local conversation cache:", err);
    }
  }, [chats]);

  // Remember the currently selected chat.
  useEffect(() => {
    if (currentId) {
      localStorage.setItem(CURRENT_KEY, currentId);
    } else {
      localStorage.removeItem(CURRENT_KEY);
    }
  }, [currentId]);

  // Auto-scroll while the user is already near the bottom.
  useEffect(() => {
    const container = messagesContainerRef.current;

    if (!container || !shouldAutoScrollRef.current) return;

    container.scrollTop = container.scrollHeight;
  }, [messages]);

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const updateCurrentMessages = useCallback(
    (chatId, nextMessages, firstQuestion = "") => {
      setChats((previous) => {
        const existing = previous.find((chat) => chat.id === chatId);

        const updated = {
          ...(existing || {
            id: chatId,
            title: titleFrom(firstQuestion),
            messages: [],
          }),
          title:
            existing?.title && existing.title !== "New conversation"
              ? existing.title
              : titleFrom(firstQuestion),
          messages: nextMessages,
          updatedAt: Date.now(),
        };

        return [
          updated,
          ...previous.filter((chat) => chat.id !== chatId),
        ];
      });
    },
    []
  );

  // Start a new conversation without deleting previous chats.
  const newChat = () => {
    if (loading) {
      abortRef.current?.abort();
      setLoading(false);
    }

    setCurrentId(makeId());
    setInput("");
    setAttachments([]);
    setError("");
    setSidebarOpen(false);
    shouldAutoScrollRef.current = true;
  };

  const openChat = (chatId) => {
    if (loading) {
      abortRef.current?.abort();
    }

    setCurrentId(chatId);
    setError("");
    setLoading(false);
    setSidebarOpen(false);
    shouldAutoScrollRef.current = true;
  };


  const MAX_FILE_SIZE = 10 * 1024 * 1024;
  const ACCEPTED_TYPES = new Set([
    "image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf",
    "text/plain", "text/markdown", "text/csv", "application/json",
    "text/x-python", "text/javascript", "text/html", "text/css",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  ]);

  const handleFileSelection = async (event) => {
    const selected = Array.from(event.target.files || []);
    event.target.value = "";
    if (!selected.length) return;
    if (attachments.length + selected.length > 5) {
      setError("Attach up to 5 files per message.");
      return;
    }
    const next = [];
    for (const file of selected) {
      if (file.size > MAX_FILE_SIZE) {
        setError(`${file.name} is larger than 10 MB.`);
        continue;
      }
      if (!ACCEPTED_TYPES.has(file.type)) {
        setError(`Unsupported file type: ${file.name}. Choose an image, PDF, text/code file, CSV, DOCX, or PPTX.`);
        continue;
      }
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
        reader.readAsDataURL(file);
      }).catch((err) => {
        setError(err.message);
        return null;
      });
      if (!dataUrl) continue;
      next.push({
        id: makeId(), name: file.name, mime_type: file.type,
        size: file.size, data_url: dataUrl,
        data: String(dataUrl).split(",")[1] || "",
      });
    }
    if (next.length) {
      setAttachments((previous) => [...previous, ...next]);
      setError("");
    }
  };

  const removeAttachment = (id) => {
    setAttachments((previous) => previous.filter((item) => item.id !== id));
  };

  const sendMessage = async (text = input) => {
    const question = text.trim();
    const filesToSend = attachments;

    if ((!question && !filesToSend.length) || loading) return;

    let chatId = currentId;

    if (!chatId) {
      const newConversation = makeChat();
      chatId = newConversation.id;

      setCurrentId(chatId);
      setChats((previous) => [newConversation, ...previous]);
    }

    const chat = chats.find((item) => item.id === chatId);
    const previousMessages = chat?.messages || [];

    const userMessage = {
      role: "user",
      content: question,
      attachments: filesToSend.map(({ id, name, mime_type, size, data_url }) => ({ id, name, mime_type, size, data_url })),
      id: makeId(),
      timestamp: Date.now(),
    };

    const assistantMessage = {
      role: "assistant",
      content: "",
      id: makeId(),
      timestamp: null,
    };

    const startingMessages = [
      ...previousMessages,
      userMessage,
      assistantMessage,
    ];

    setInput("");
    setAttachments([]);
    setError("");
    setLoading(true);
    shouldAutoScrollRef.current = true;

    updateCurrentMessages(chatId, startingMessages, question || filesToSend[0]?.name || "File question");

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
        body: JSON.stringify({
          message: question,
          attachments: filesToSend.map(({ name, mime_type, data }) => ({ name, mime_type, data })),
          user_id: userId,
          session_id: chatId,
          session_name: titleFrom(question || filesToSend[0]?.name || "File question"),
          history: previousMessages.map(({ role, content }) => ({
            role,
            content,
          })),
        }),
      });

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;

        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch {
          // Keep the default error message.
        }

        throw new Error(detail);
      }

      if (!response.body) {
        throw new Error(
          "Your browser does not support streaming responses."
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let assistantText = "";

      const applyEvent = (line) => {
        if (!line.trim()) return;

        let event;

        try {
          event = JSON.parse(line);
        } catch {
          return;
        }

        if (event.error) {
          throw new Error(event.error);
        }

        if (event.token) {
          assistantText += event.token;

          const nextMessages = [
            ...startingMessages.slice(0, -1),
            {
              ...assistantMessage,
              content: assistantText,
              timestamp: assistantMessage.timestamp || Date.now(),
            },
          ];

          updateCurrentMessages(chatId, nextMessages, question || filesToSend[0]?.name || "File question");
        }
      };

      while (true) {
        const { value, done } = await reader.read();

        buffer += decoder.decode(
          value || new Uint8Array(),
          { stream: !done }
        );

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          applyEvent(line);
        }

        if (done) break;
      }

      if (buffer.trim()) {
        applyEvent(buffer);
      }

      if (!assistantText.trim()) {
        const fallback =
          "I couldn't generate a reply. Please check the Gemini connection and try again.";

        updateCurrentMessages(
          chatId,
          [
            ...startingMessages.slice(0, -1),
            {
              ...assistantMessage,
              content: fallback,
              timestamp: Date.now(),
            },
          ],
          question
        );
      }

      // Reload the persisted record after the backend finishes saving the response.
      loadDatabaseChats(chatId).catch((err) => {
        console.error("Chat replied, but history refresh failed:", err);
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(
          err.message || "Something went wrong. Please try again."
        );

        const failedMessages = [
          ...startingMessages.slice(0, -1),
          {
            ...assistantMessage,
            content: "",
          },
        ].filter(
          (message) =>
            message.role !== "assistant" || message.content
        );

        updateCurrentMessages(
          chatId,
          failedMessages,
          question
        );
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    sendMessage();
  };

  const suggestions = [
    {
      icon: "♡",
      text: "Tell me something comforting",
    },
    {
      icon: "✧",
      text: "Explain a topic I'm studying",
    },
    {
      icon: "☻",
      text: "Help me plan my study time",
    },
  ];

  return (
    <div className={`app ${darkMode ? "theme-dark" : "theme-light"}`}>
      {/* Floating decorative hearts */}
      <div className="floating-hearts" aria-hidden="true">
        {Array.from({ length: 18 }, (_, index) => (
          <span
            key={index}
            className={`floating-heart heart-${index % 6}`}
            style={{
              left: `${(index * 37 + 8) % 100}%`,
              animationDelay: `${(index % 9) * -2.3}s`,
              animationDuration: `${14 + (index % 7) * 2}s`,
              fontSize: `${12 + (index % 5) * 7}px`,
            }}
          >
            {index % 3 === 0 ? "♥" : "♡"}
          </span>
        ))}
      </div>

      {sidebarOpen && (
        <button
          className="mobile-scrim"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}
      >
        <div className="brand-row">
          <div className="brand-mark">♡</div>

          <div>
            <div className="brand-name">MyGPT</div>
            <div className="brand-caption">
              your little safe space
            </div>
          </div>

          <button
            className="icon-button mobile-close"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            ×
          </button>
        </div>

        <button className="new-chat-button" onClick={newChat}>
          <span>＋</span> New chat
        </button>

        <div className="sidebar-label" />

        <button
          className="nav-item active"
          onClick={() => setSidebarOpen(false)}
        >
          <span>⌂</span> Conversations
        </button>

        <div className="sidebar-label recent-label">
          RECENT CHATS
        </div>

        <div className="recent-list">
          {sortedChats.length === 0 ? (
            <p className="empty-recent">
              Your conversations will appear here.
            </p>
          ) : (
            sortedChats.map((chat) => (
              <button
                key={chat.id}
                className={`recent-chat ${
                  chat.id === currentId ? "selected" : ""
                }`}
                onClick={() => openChat(chat.id)}
                title={chat.title}
              >
                <span className="recent-chat-icon">◌</span>

                <span className="recent-chat-title" style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {chat.title || "New conversation"}
                  </span>
                  <span style={{ fontSize: "0.7rem", opacity: 0.68, marginTop: 3 }}>
                    {formatDateTime(chat.updatedAt)}
                  </span>
                </span>

              </button>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="note-heart">♡</span>
            <p>
              I am Always with you Sadhu Maah Dont feel About
              anything, call me if you feel very low I am Always
              For You.
            </p>
          </div>

          <button
            className="theme-toggle"
            onClick={() => setDarkMode((value) => !value)}
          >
            <span>{darkMode ? "☼" : "☾"}</span>
            {darkMode
              ? "Switch to light mode"
              : "Switch to dark mode"}
          </button>

          <div className="local-label">
            <span className="local-dot" />
            Runs with Love
          </div>
        </div>
      </aside>

      {/* Main chat panel */}
      {/* Keep this panel transparent so the floating hearts behind it remain visible. */}
      <main className="main-panel" style={{ background: "transparent" }}>
        <header className="topbar">
          <button
            className="icon-button menu-button"
            aria-label="Open sidebar"
            onClick={() => setSidebarOpen(true)}
          >
            ☰
          </button>

          <div className="topbar-title">
            <span className="status-dot" />
            A little space for you
          </div>

          <button className="topbar-new" onClick={newChat}>
            ＋ New
          </button>
        </header>

        <section
          className={`chat-area ${
            messages.length ? "has-messages" : "welcome-state"
          }`}
        >
          {messages.length === 0 ? (
            <div className="welcome-content">
              <div className="welcome-symbol" aria-hidden="true">
                <span className="spark spark-one">✦</span>
                <span className="welcome-heart">♡</span>
                <span className="spark spark-two">✧</span>
              </div>

              <p className="eyebrow">
                A LITTLE SPACE JUST FOR YOU
              </p>

              <h1>
                Hi there, lovely <span>♡</span>
              </h1>

              <p className="welcome-copy">
                What's on your mind today? We can figure things
                out together, one little step at a time.
              </p>

              <div className="suggestion-row">
                {suggestions.map((suggestion) => (
                  <button
                    key={suggestion.text}
                    className="suggestion-chip"
                    onClick={() => sendMessage(suggestion.text)}
                  >
                    <span>{suggestion.icon}</span>
                    {suggestion.text}
                  </button>
                ))}
              </div>

              <div className="welcome-footnote" />
            </div>
          ) : (
            <div
              className="messages-list"
              ref={messagesContainerRef}
              onScroll={(event) => {
                const element = event.currentTarget;

                const distanceFromBottom =
                  element.scrollHeight -
                  element.scrollTop -
                  element.clientHeight;

                shouldAutoScrollRef.current =
                  distanceFromBottom < 100;
              }}
            >
              <div className="conversation-heading">
                <div>
                  <h2>Your conversation</h2>
                  <p>
                    Ask questions, explore ideas, and learn at your
                    own pace.
                  </p>
                </div>

                <button
                  className="heading-new"
                  onClick={newChat}
                >
                  ＋ New
                </button>
              </div>

              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message-row ${
                    message.role === "user"
                      ? "user-row"
                      : "assistant-row"
                  }`}
                >
                  {message.role === "assistant" && (
                    <div className="message-avatar">♡</div>
                  )}

                  <div
                    className={`message-bubble ${
                      message.role === "user"
                        ? "user-bubble"
                        : "assistant-bubble"
                    }`}
                  >
                    {message.attachments?.length > 0 && (
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: message.content ? 8 : 0 }}>
                        {message.attachments.map((file) => (
                          <div key={file.id || file.name} style={{ maxWidth: 240, border: "1px solid var(--border-color, rgba(255,255,255,.18))", borderRadius: 12, padding: 8, fontSize: "0.82rem" }}>
                            {file.mime_type?.startsWith("image/") && file.data_url ? (
                              <img src={file.data_url} alt={file.name} style={{ display: "block", maxWidth: 200, maxHeight: 160, objectFit: "contain", borderRadius: 8, marginBottom: 5 }} />
                            ) : <span aria-hidden="true">📎 </span>}
                            <span style={{ overflowWrap: "anywhere" }}>{file.name}</span>
                          </div>
                        ))}
                      </div>
                    )}
                    {message.content ? (
                      <div
                        className={`message-text ${
                          message.role === "assistant"
                            ? "markdown-content"
                            : ""
                        }`}
                      >
                        {message.role === "assistant" ? (
                          <TypingMarkdown text={message.content} speed={18} />
                        ) : (
                          message.content
                        )}
                      </div>
                    ) : loading &&
                      message.role === "assistant" ? (
                      <span className="typing">
                        <i />
                        <i />
                        <i />
                      </span>
                    ) : null}
                    {message.timestamp && (
                      <div style={{ fontSize: "0.68rem", opacity: 0.65, marginTop: 6, textAlign: message.role === "user" ? "right" : "left" }}>
                        {formatDateTime(message.timestamp)}
                      </div>
                    )}
                  </div>

                  {message.role === "user" && (
                    <div className="user-avatar">☺</div>
                  )}
                </div>
              ))}

              {loading && (
                <div className="streaming-label">
                  MyGPT is thinking gently…
                </div>
              )}
            </div>
          )}
        </section>

        {/* Message composer */}
        <footer className="composer-wrap">
          {error && (
            <div className="error-message">
              <span>!</span>
              {error}

              <button
                onClick={() => setError("")}
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}

          <form className="composer" onSubmit={handleSubmit}>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*,.pdf,.txt,.md,.csv,.json,.py,.js,.html,.css,.docx,.pptx"
              onChange={handleFileSelection}
              style={{ display: "none" }}
            />
            {attachments.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, padding: "8px 12px 0" }}>
                {attachments.map((file) => (
                  <div key={file.id} style={{ display: "flex", alignItems: "center", gap: 8, borderRadius: 10, padding: "7px 10px", background: "rgba(150,120,180,.16)", maxWidth: "100%" }}>
                    {file.mime_type.startsWith("image/") ? (
                      <img src={file.data_url} alt={file.name} style={{ width: 42, height: 42, objectFit: "cover", borderRadius: 6 }} />
                    ) : <span>📎</span>}
                    <span style={{ fontSize: "0.8rem", overflowWrap: "anywhere" }}>{file.name}</span>
                    <button type="button" onClick={() => removeAttachment(file.id)} aria-label={`Remove ${file.name}`} style={{ border: 0, background: "transparent", cursor: "pointer", fontSize: "1.1rem" }}>×</button>
                  </div>
                ))}
              </div>
            )}
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Tell me what's on your mind…"
              rows={1}
              disabled={loading}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey
                ) {
                  event.preventDefault();
                  handleSubmit(event);
                }
              }}
            />

            <div className="composer-bottom">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={loading || attachments.length >= 5}
                aria-label="Attach image or file"
                title="Attach image or file"
                style={{ border: "1px solid var(--border-color, rgba(255,255,255,.2))", borderRadius: 12, padding: "8px 11px", background: "transparent", color: "inherit", cursor: "pointer", opacity: loading || attachments.length >= 5 ? 0.5 : 1 }}
              >
                ＋ Attach
              </button>
              <span className="composer-hint">
                ♡ &nbsp; You're allowed to take your time.
              </span>

              <button
                className="send-button"
                type="submit"
                disabled={loading || (!input.trim() && attachments.length === 0)}
                aria-label="Send message"
              >
                {loading ? (
                  <span className="send-spinner" />
                ) : (
                  <span>↑</span>
                )}
              </button>
            </div>
          </form>

          <p className="privacy-note">
            MyGPT is Your GPT.
          </p>
        </footer>
      </main>
    </div>
  );
}
