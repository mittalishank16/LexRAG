"use client";
// app/chat/page.tsx  —  LexRAG Legal Chat Interface
// Full-featured chat with file upload, streaming responses,
// strategy badge, context viewer, and conversation history

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ──────────────────────────────────────────────────────────────────
type Strategy = "LEGAL" | "DOCUMENT" | "BOTH";

interface Message {
  id:        string;
  role:      "user" | "assistant";
  content:   string;
  strategy?: Strategy;
  critique?: string;
  timestamp: Date;
}

// ── Strategy badge colours ─────────────────────────────────────────────────
const STRATEGY_STYLES: Record<Strategy, { bg: string; text: string; label: string }> = {
  LEGAL:    { bg: "bg-blue-500/15",  text: "text-blue-300",  label: "⚖️  Legal KB"    },
  DOCUMENT: { bg: "bg-amber-500/15", text: "text-amber-300", label: "📄 Your Doc"    },
  BOTH:     { bg: "bg-purple-500/15",text: "text-purple-300",label: "🔀 Legal + Doc" },
};

// ── Suggested questions ────────────────────────────────────────────────────
const SUGGESTIONS = [
  "What does Article 21 of the Indian Constitution guarantee?",
  "What is the punishment for murder under Section 302 IPC?",
  "Explain the right against self-incrimination under Article 20.",
  "What are the fundamental rights guaranteed under Part III?",
  "What is habeas corpus and when can it be invoked?",
  "What is the difference between bailable and non-bailable offences?",
];

// ── Helpers ────────────────────────────────────────────────────────────────
function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function formatTime(d: Date) {
  return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

// ════════════════════════════════════════════════════════════════════════════
export default function ChatPage() {
  const [messages,    setMessages]    = useState<Message[]>([]);
  const [input,       setInput]       = useState("");
  const [loading,     setLoading]     = useState(false);
  const [filePath,    setFilePath]    = useState<string | null>(null);
  const [fileName,    setFileName]    = useState<string | null>(null);
  const [uploading,   setUploading]   = useState(false);
  const [showContext, setShowContext] = useState<string | null>(null);

  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const fileRef    = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom on new message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ── File upload ──────────────────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Only PDF files are supported.");
      return;
    }
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setFilePath(res.data.file_path);
      setFileName(file.name);

      // System message acknowledging upload
      setMessages(prev => [...prev, {
        id:        uid(),
        role:      "assistant",
        content:   `Document **${file.name}** loaded successfully. I can now answer questions combining this document with the Indian legal knowledge base. What would you like to know?`,
        timestamp: new Date(),
      }]);
    } catch (err) {
      alert("Upload failed. Make sure the backend is running.");
    } finally {
      setUploading(false);
    }
  };

  // ── Send message ─────────────────────────────────────────────────────────
  const send = async (question: string = input.trim()) => {
    if (!question || loading) return;

    const userMsg: Message = {
      id:        uid(),
      role:      "user",
      content:   question,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post(`${API}/chat`, {
        question,
        file_path: filePath ?? null,
      });

      const assistantMsg: Message = {
        id:        uid(),
        role:      "assistant",
        content:   res.data.answer,
        strategy:  res.data.strategy as Strategy,
        critique:  res.data.critique,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages(prev => [...prev, {
        id:        uid(),
        role:      "assistant",
        content:   `Error: ${err?.response?.data?.detail ?? "Backend unreachable. Make sure the FastAPI server is running."}`,
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  // ── Keyboard handler ─────────────────────────────────────────────────────
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  // ── Clear conversation ───────────────────────────────────────────────────
  const clearChat = () => {
    setMessages([]);
    setFilePath(null);
    setFileName(null);
  };

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-screen bg-[#0A0F1E] text-white">

      {/* ── Background grid ───────────────────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none opacity-30"
        style={{
          backgroundImage: `linear-gradient(rgba(46,117,182,0.04) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(46,117,182,0.04) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }} />

      {/* ── Top nav ───────────────────────────────────────────────────────── */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#0A0F1E]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <div className="w-7 h-7 rounded-md flex items-center justify-center text-sm"
              style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>⚖</div>
            <span className="font-semibold text-white/80 text-sm" style={{ fontFamily: "'Georgia', serif" }}>LexRAG</span>
          </Link>
          <span className="text-white/20 text-sm">/</span>
          <span className="text-sm text-white/50">Legal Chat</span>
        </div>

        <div className="flex items-center gap-3">
          {/* Document status pill */}
          {fileName ? (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-amber-500/30 bg-amber-500/10">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
              <span className="text-xs text-amber-300 max-w-[140px] truncate">{fileName}</span>
              <button onClick={() => { setFilePath(null); setFileName(null); }}
                className="text-amber-400/60 hover:text-amber-400 transition-colors text-xs ml-1">✕</button>
            </div>
          ) : (
            <button onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 text-white/50 hover:border-white/20 hover:text-white/70 transition-all text-xs">
              {uploading ? (
                <><span className="w-3 h-3 border border-white/30 border-t-white/80 rounded-full animate-spin" />Uploading...</>
              ) : (
                <><span>📎</span>Upload PDF</>
              )}
            </button>
          )}

          <input ref={fileRef} type="file" accept=".pdf" className="hidden"
            onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} />

          {messages.length > 0 && (
            <button onClick={clearChat}
              className="text-xs text-white/30 hover:text-white/60 transition-colors px-2 py-1">
              Clear
            </button>
          )}

          <Link href="/contracts"
            className="text-xs px-3 py-1.5 rounded-lg border border-white/10 text-white/50 hover:text-white/70 hover:border-white/20 transition-all">
            Contracts →
          </Link>
        </div>
      </header>

      {/* ── Chat area ─────────────────────────────────────────────────────── */}
      <div className="relative z-10 flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">

          {/* Empty state */}
          {isEmpty && (
            <div className="text-center pt-12">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-5 text-3xl"
                style={{ background: "linear-gradient(135deg, #1F3864, #2E75B6)" }}>
                ⚖️
              </div>
              <h2 className="text-2xl font-semibold text-white/80 mb-3"
                style={{ fontFamily: "'Georgia', serif" }}>
                Ask Anything About Indian Law
              </h2>
              <p className="text-sm text-white/40 max-w-md mx-auto leading-relaxed mb-8">
                Powered by InLegalBERT and a 7-node LangGraph agent. Searches the Constitution,
                IPC, CrPC, Evidence Act and more. Upload a PDF to include your own document.
              </p>

              {/* Suggestion pills */}
              <div className="grid grid-cols-2 gap-2 max-w-2xl mx-auto text-left">
                {SUGGESTIONS.map(s => (
                  <button key={s}
                    onClick={() => send(s)}
                    className="p-3.5 rounded-xl border border-white/8 bg-white/[0.03] hover:border-[#2E75B6]/40 hover:bg-white/[0.05] transition-all text-left">
                    <p className="text-xs text-white/55 leading-relaxed">{s}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Messages */}
          {messages.map(msg => (
            <div key={msg.id} className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>

              {/* Avatar */}
              <div className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center text-sm
                ${msg.role === "user"
                  ? "bg-white/10 text-white/60"
                  : "bg-gradient-to-br from-[#1F3864] to-[#2E75B6] text-white"}`}>
                {msg.role === "user" ? "U" : "⚖"}
              </div>

              {/* Bubble */}
              <div className={`max-w-[80%] ${msg.role === "user" ? "items-end" : "items-start"} flex flex-col gap-1.5`}>

                {/* Strategy badge for assistant */}
                {msg.role === "assistant" && msg.strategy && STRATEGY_STYLES[msg.strategy] && (
                  <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium
                    ${STRATEGY_STYLES[msg.strategy].bg} ${STRATEGY_STYLES[msg.strategy].text}`}>
                    {STRATEGY_STYLES[msg.strategy].label}
                  </div>
                )}

                {/* Content bubble */}
                <div className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap
                  ${msg.role === "user"
                    ? "bg-[#2E75B6]/20 text-white/85 rounded-tr-sm border border-[#2E75B6]/20"
                    : "bg-white/[0.04] text-white/80 rounded-tl-sm border border-white/6"}`}>
                  {msg.content}
                </div>

                {/* Critique toggle */}
                {msg.critique && (
                  <button
                    onClick={() => setShowContext(showContext === msg.id ? null : msg.id)}
                    className="text-xs text-white/25 hover:text-white/50 transition-colors px-1">
                    {showContext === msg.id ? "Hide evaluation ↑" : "View evaluation ↓"}
                  </button>
                )}

                {showContext === msg.id && msg.critique && (
                  <div className="px-3 py-2.5 rounded-xl bg-white/[0.03] border border-white/6 text-xs text-white/40 leading-relaxed font-mono max-w-full">
                    {msg.critique}
                  </div>
                )}

                <span className="text-[10px] text-white/20 px-1">
                  {formatTime(msg.timestamp)}
                </span>
              </div>
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg shrink-0 flex items-center justify-center text-sm bg-gradient-to-br from-[#1F3864] to-[#2E75B6]">
                ⚖
              </div>
              <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-white/[0.04] border border-white/6">
                <div className="flex gap-1 items-center">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#2E75B6]/60 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-[#2E75B6]/60 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-[#2E75B6]/60 animate-bounce" style={{ animationDelay: "300ms" }} />
                  <span className="text-xs text-white/30 ml-2">Searching legal knowledge base…</span>
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ── Input bar ─────────────────────────────────────────────────────── */}
      <div className="relative z-10 border-t border-white/5 bg-[#0A0F1E]/90 backdrop-blur-sm px-4 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex gap-3 items-end p-3 rounded-2xl border border-white/8 bg-white/[0.03] focus-within:border-[#2E75B6]/40 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask any Indian law question… (Enter to send, Shift+Enter for new line)"
              rows={1}
              className="flex-1 bg-transparent text-sm text-white/80 placeholder-white/20 resize-none outline-none leading-relaxed max-h-36 overflow-y-auto"
              style={{ scrollbarWidth: "none" }}
            />
            <button
              onClick={() => send()}
              disabled={!input.trim() || loading}
              className="shrink-0 w-9 h-9 rounded-xl flex items-center justify-center transition-all disabled:opacity-30 disabled:cursor-not-allowed hover:opacity-90"
              style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 7h12M7 1l6 6-6 6" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
          <p className="text-[10px] text-white/20 text-center mt-2">
            LexRAG can make mistakes. Verify important legal information with a qualified lawyer.
          </p>
        </div>
      </div>
    </div>
  );
}
