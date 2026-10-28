import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const CHATS_KEY = "mygpt_saved_chats_v1";
const CURRENT_KEY = "mygpt_current_chat_v1";

const makeId = () => `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
const makeChat = () => ({ id: makeId(), title: "New conversation", updatedAt: Date.now(), messages: [] });
const titleFrom = (text) => {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > 36 ? `${clean.slice(0, 36)}…` : clean || "New conversation";
};

function readChats() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHATS_KEY) || "[]");
    return Array.isArray(parsed) ? parsed.filter((chat) => chat && chat.id && Array.isArray(chat.messages)) : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [chats, setChats] = useState(() => readChats());
  const [currentId, setCurrentId] = useState(() => localStorage.getItem(CURRENT_KEY) || "");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(true);
  const [error, setError] = useState("");
  const messagesContainerRef = useRef(null);
  const shouldAutoScrollRef = useRef(true);
  const abortRef = useRef(null);

  const sortedChats = useMemo(
    () => [...chats].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)),
    [chats]
  );
  const currentChat = chats.find((chat) => chat.id === currentId) || null;
  const messages = currentChat?.messages || [];

  useEffect(() => {
    localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    if (currentId) localStorage.setItem(CURRENT_KEY, currentId);
    else localStorage.removeItem(CURRENT_KEY);
  }, [currentId]);

  // Keep streaming stable: follow new tokens only when the reader is already near the bottom.
  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container || !shouldAutoScrollRef.current) return;
    container.scrollTop = container.scrollHeight;
  }, [messages]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const updateCurrentMessages = useCallback((chatId, nextMessages, firstQuestion = "") => {
    setChats((previous) => {
      const exists = previous.some((chat) => chat.id === chatId);
      const existing = previous.find((chat) => chat.id === chatId);
      const updated = {
        ...(existing || { id: chatId, title: titleFrom(firstQuestion), messages: [] }),
        title: existing?.title && existing.title !== "New conversation" ? existing.title : titleFrom(firstQuestion),
        messages: nextMessages,
        updatedAt: Date.now(),
      };
      return [updated, ...previous.filter((chat) => chat.id !== chatId && chat.id)].filter(Boolean);
    });
  }, []);

  const newChat = () => {
  // If the current conversation is empty, don't create another chat.
  const hasConversation = messages.some(
    (message) => message.content?.trim().length > 0
  );

  if (!hasConversation) {
    setInput("");
    return;
  }

  // Save the current conversation before starting another one.
  const currentChat = {
    id: currentId,
    title:
      messages.find((message) => message.role === "user")?.content
        ?.trim()
        .slice(0, 40) || "New Chat",
    messages,
    updatedAt: Date.now(),
  };

  setChats((previousChats) => {
    const updatedChats = [
      currentChat,
      ...previousChats.filter((chat) => chat.id !== currentId),
    ];

    localStorage.setItem(CHATS_KEY, JSON.stringify(updatedChats));
    return updatedChats;
  });

  // Start a new conversation.
  setCurrentId(crypto.randomUUID());
  setMessages([]);
  setInput("");
  setActiveNav("Chat");
};

  const openChat = (chatId) => {
    if (loading) abortRef.current?.abort();
    setCurrentId(chatId);
    setError("");
    setLoading(false);
    setSidebarOpen(false);
  };

  const deleteChat = (event, chatId) => {
    event.stopPropagation();
    setChats((previous) => previous.filter((chat) => chat.id !== chatId));
    if (currentId === chatId) setCurrentId("");
  };

  const sendMessage = async (text = input) => {
    const question = text.trim();
    if (!question || loading) return;

    let chatId = currentId;
    if (!chatId) {
      const newConversation = makeChat();
      chatId = newConversation.id;
      setCurrentId(chatId);
      setChats((previous) => [newConversation, ...previous]);
    }

    const chat = chats.find((item) => item.id === chatId);
    const previousMessages = chat?.messages || [];
    const userMessage = { role: "user", content: question, id: makeId() };
    const assistantMessage = { role: "assistant", content: "", id: makeId() };
    const startingMessages = [...previousMessages, userMessage, assistantMessage];

    setInput("");
    setError("");
    setLoading(true);
    updateCurrentMessages(chatId, startingMessages, question);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          message: question,
          history: previousMessages.map(({ role, content }) => ({ role, content })),
        }),
      });
      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch { /* keep default */ }
        throw new Error(detail);
      }
      if (!response.body) throw new Error("Your browser does not support streaming responses.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let assistantText = "";

      const applyEvent = (line) => {
        if (!line.trim()) return;
        let event;
        try { event = JSON.parse(line); } catch { return; }
        if (event.error) throw new Error(event.error);
        if (event.token) {
          assistantText += event.token;
          const nextMessages = [...startingMessages.slice(0, -1), { ...assistantMessage, content: assistantText }];
          updateCurrentMessages(chatId, nextMessages, question);
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) applyEvent(line);
        if (done) break;
      }
      if (buffer.trim()) applyEvent(buffer);
      if (!assistantText.trim()) {
        const fallback = "I couldn't generate a reply. Please check that Ollama is running and try again.";
        updateCurrentMessages(chatId, [...startingMessages.slice(0, -1), { ...assistantMessage, content: fallback }], question);
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(err.message || "Something went wrong. Please try again.");
        const failedMessages = [...startingMessages.slice(0, -1), { ...assistantMessage, content: "" }].filter((message) => message.role !== "assistant" || message.content);
        updateCurrentMessages(chatId, failedMessages, question);
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
    { icon: "♡", text: "Tell me something comforting" },
    { icon: "✧", text: "Explain a topic I'm studying" },
    { icon: "☻", text: "Help me plan my study time" },
  ];

  return (
    <div className={`app ${darkMode ? "theme-dark" : "theme-light"}`}>
      {sidebarOpen && <button className="mobile-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />}

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark">♡</div>
          <div><div className="brand-name">MyGPT</div><div className="brand-caption">your little safe space</div></div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)} aria-label="Close sidebar">×</button>
        </div>

        <button className="new-chat-button" onClick={newChat}><span>＋</span> New chat</button>
        <div className="sidebar-label"></div>
        <button className="nav-item active" onClick={() => { setSidebarOpen(false); }}><span>⌂</span> Conversations</button>
        <div className="sidebar-label recent-label">RECENT CHATS</div>
        <div className="recent-list">
          {sortedChats.length === 0 ? (
            <p className="empty-recent">Your conversations will appear here.</p>
          ) : sortedChats.map((chat) => (
            <button key={chat.id} className={`recent-chat ${chat.id === currentId ? "selected" : ""}`} onClick={() => openChat(chat.id)} title={chat.title}>
              <span className="recent-chat-icon">◌</span><span className="recent-chat-title">{chat.title || "New conversation"}</span>
              <span className="delete-chat" role="button" tabIndex={0} aria-label={`Delete ${chat.title}`} onClick={(event) => deleteChat(event, chat.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") deleteChat(event, chat.id); }}>×</span>
            </button>
          ))}
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-note"><span className="note-heart">♡</span><p>I am Always with you Sadhu Maah Dont feel About anything ,call me if you feel very low I am Always For You.</p></div>
          <button className="theme-toggle" onClick={() => setDarkMode((value) => !value)}><span>{darkMode ? "☼" : "☾"}</span>{darkMode ? "Switch to light mode" : "Switch to dark mode"}</button>
          <div className="local-label"><span className="local-dot" /> Runs with Love</div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button menu-button" aria-label="Open sidebar" onClick={() => setSidebarOpen(true)}>☰</button>
          <div className="topbar-title"><span className="status-dot" /> A little space for you</div>
          <button className="topbar-new" onClick={newChat}>＋ New</button>
        </header>

        <section className={`chat-area ${messages.length ? "has-messages" : "welcome-state"}`}>
          {messages.length === 0 ? (
            <div className="welcome-content">
              <div className="welcome-symbol" aria-hidden="true"><span className="spark spark-one">✦</span><span className="welcome-heart">♡</span><span className="spark spark-two">✧</span></div>
              <p className="eyebrow">A LITTLE SPACE JUST FOR YOU</p>
              <h1>Hi there, lovely <span>♡</span></h1>
              <p className="welcome-copy">What's on your mind today? We can figure things out together, one little step at a time.</p>
              <div className="suggestion-row">
                {suggestions.map((suggestion) => <button key={suggestion.text} className="suggestion-chip" onClick={() => sendMessage(suggestion.text)}><span>{suggestion.icon}</span>{suggestion.text}</button>)}
              </div>
              <div className="welcome-footnote"></div>
            </div>
          ) : (
            <div
              className="messages-list"
              ref={messagesContainerRef}
              onScroll={(event) => {
                const element = event.currentTarget;
                const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight;
                shouldAutoScrollRef.current = distanceFromBottom < 100;
              }}
            >
              <div className="conversation-heading"><div><h2>Your conversation</h2><p>Ask questions, explore ideas, and learn at your own pace.</p></div><button className="heading-new" onClick={newChat}>＋ New</button></div>
              {messages.map((message) => (
                <div key={message.id} className={`message-row ${message.role === "user" ? "user-row" : "assistant-row"}`}>
                  {message.role === "assistant" && <div className="message-avatar">♡</div>}
                  <div className={`message-bubble ${message.role === "user" ? "user-bubble" : "assistant-bubble"}`}>
                    {message.content ? <div className="message-text">{message.content}</div> : loading && message.role === "assistant" ? <span className="typing"><i /><i /><i /></span> : null}
                  </div>
                  {message.role === "user" && <div className="user-avatar">☺</div>}
                </div>
              ))}
              {loading && <div className="streaming-label">MyGPT is thinking gently…</div>}
            </div>
          )}
        </section>

        <footer className="composer-wrap">
          {error && <div className="error-message"><span>!</span>{error}<button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
          <form className="composer" onSubmit={handleSubmit}>
            <textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder="Tell me what's on your mind…" rows={1} disabled={loading} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); handleSubmit(event); } }} />
            <div className="composer-bottom"><span className="composer-hint">♡ &nbsp; You're allowed to take your time.</span><button className="send-button" type="submit" disabled={loading || !input.trim()} aria-label="Send message">{loading ? <span className="send-spinner" /> : <span>↑</span>}</button></div>
          </form>
          <p className="privacy-note">MyGPT is Your GPT.</p>
        </footer>
      </main>
    </div>
  );
}
