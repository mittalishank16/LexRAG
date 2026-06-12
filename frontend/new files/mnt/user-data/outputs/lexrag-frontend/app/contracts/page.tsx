"use client";
// app/contracts/page.tsx  —  LexRAG Contract Intelligence
// Upload contracts → AI extracts metadata → View stored contracts
// → Query contracts in natural language → Renewal status dashboard

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import axios from "axios";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────
interface Contract {
  id:                  string;
  filename:            string;
  party_a:             string;
  party_b:             string;
  contract_type:       string;
  start_date:          string | null;
  end_date:            string | null;
  renewal_date:        string | null;
  notice_period_days:  number;
  auto_renewal:        boolean;
  governing_law:       string;
  key_obligations:     string;
  reminder_sent_30:    boolean;
  reminder_sent_7:     boolean;
  reminder_sent_1:     boolean;
  created_at:          string;
}

// ── Helpers ───────────────────────────────────────────────────────────────
function daysUntil(dateStr: string | null): number | null {
  if (!dateStr) return null;
  const diff = new Date(dateStr).getTime() - Date.now();
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

function urgencyClass(days: number | null): string {
  if (days === null)   return "text-white/30";
  if (days <= 1)       return "text-red-400";
  if (days <= 7)       return "text-orange-400";
  if (days <= 30)      return "text-amber-400";
  return "text-green-400";
}

function urgencyBadge(days: number | null): { bg: string; label: string } {
  if (days === null)   return { bg: "bg-white/5 border-white/10",             label: "No date" };
  if (days <= 1)       return { bg: "bg-red-500/15 border-red-500/30",        label: `${days}d left` };
  if (days <= 7)       return { bg: "bg-orange-500/15 border-orange-500/30",  label: `${days}d left` };
  if (days <= 30)      return { bg: "bg-amber-500/15 border-amber-500/30",    label: `${days}d left` };
  return                      { bg: "bg-green-500/15 border-green-500/30",    label: `${days}d left` };
}

function formatDate(d: string | null): string {
  if (!d) return "—";
  return new Date(d).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
}

// ════════════════════════════════════════════════════════════════════════════
export default function ContractsPage() {
  const [contracts,  setContracts]  = useState<Contract[]>([]);
  const [loading,    setLoading]    = useState(false);
  const [fetching,   setFetching]   = useState(true);
  const [uploading,  setUploading]  = useState(false);
  const [selected,   setSelected]   = useState<Contract | null>(null);
  const [query,      setQuery]      = useState("");
  const [queryResp,  setQueryResp]  = useState("");
  const [querying,   setQuerying]   = useState(false);
  const [uploadMsg,  setUploadMsg]  = useState("");
  const [dragOver,   setDragOver]   = useState(false);
  const [userId]                    = useState("default");

  const fileRef = useRef<HTMLInputElement>(null);

  // ── Fetch contracts on mount ─────────────────────────────────────────────
  useEffect(() => {
    fetchContracts();
  }, []);

  const fetchContracts = async () => {
    setFetching(true);
    try {
      const res = await axios.get(`${API}/contracts?user_id=${userId}`);
      setContracts(res.data ?? []);
    } catch {
      console.error("Failed to fetch contracts");
    } finally {
      setFetching(false);
    }
  };

  // ── Upload contract PDF ──────────────────────────────────────────────────
  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadMsg("❌ Only PDF files are supported.");
      return;
    }
    setUploading(true);
    setUploadMsg("🔍 Extracting contract metadata with AI…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("user_id", userId);
      await axios.post(`${API}/contracts/analyze`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadMsg("✅ Contract saved! Reminder emails scheduled. Refreshing…");
      await fetchContracts();
      setTimeout(() => setUploadMsg(""), 4000);
    } catch (err: any) {
      setUploadMsg(`❌ ${err?.response?.data?.detail ?? "Upload failed. Check backend."}`);
    } finally {
      setUploading(false);
    }
  };

  // ── Drag and drop ────────────────────────────────────────────────────────
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  // ── Natural language query ───────────────────────────────────────────────
  const handleQuery = async () => {
    if (!query.trim() || querying) return;
    setQuerying(true);
    setQueryResp("");
    try {
      const res = await axios.post(`${API}/chat`, {
        question: `[Contract Query] ${query}`,
        file_path: null,
      });
      setQueryResp(res.data.answer);
    } catch {
      setQueryResp("Failed to query. Make sure the backend is running.");
    } finally {
      setQuerying(false);
    }
  };

  // ── Derived stats ────────────────────────────────────────────────────────
  const expiringSoon = contracts.filter(c => {
    const d = daysUntil(c.renewal_date);
    return d !== null && d <= 30 && d >= 0;
  }).length;

  const expired = contracts.filter(c => {
    const d = daysUntil(c.renewal_date);
    return d !== null && d < 0;
  }).length;

  // ────────────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-[#0A0F1E] text-white">

      {/* ── Background ────────────────────────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `linear-gradient(rgba(46,117,182,0.03) 1px, transparent 1px),
                            linear-gradient(90deg, rgba(46,117,182,0.03) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }} />
      <div className="fixed top-0 right-0 w-[400px] h-[400px] rounded-full pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(39,174,96,0.06) 0%, transparent 70%)" }} />

      {/* ── Nav ───────────────────────────────────────────────────────────── */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4 border-b border-white/5 bg-[#0A0F1E]/80 backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <div className="w-7 h-7 rounded-md flex items-center justify-center text-sm"
              style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>⚖</div>
            <span className="font-semibold text-white/80 text-sm" style={{ fontFamily: "'Georgia', serif" }}>LexRAG</span>
          </Link>
          <span className="text-white/20 text-sm">/</span>
          <span className="text-sm text-white/50">Contract Intelligence</span>
        </div>
        <Link href="/chat"
          className="text-xs px-3 py-1.5 rounded-lg border border-white/10 text-white/50 hover:text-white/70 hover:border-white/20 transition-all">
          Legal Chat →
        </Link>
      </header>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-8 space-y-8">

        {/* ── Stats row ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { value: contracts.length,           label: "Total Contracts",  color: "text-white/70"    },
            { value: expiringSoon,                label: "Expiring ≤ 30d",  color: "text-amber-400"   },
            { value: expired,                     label: "Expired",         color: "text-red-400"     },
            { value: contracts.filter(c=>c.auto_renewal).length, label: "Auto-Renewal", color: "text-green-400" },
          ].map(({ value, label, color }) => (
            <div key={label} className="p-5 rounded-xl border border-white/8 bg-white/[0.03]">
              <div className={`text-3xl font-bold mb-1 ${color}`}
                style={{ fontFamily: "'Georgia', serif" }}>{value}</div>
              <div className="text-xs text-white/35 uppercase tracking-wide">{label}</div>
            </div>
          ))}
        </div>

        {/* ── Upload zone ────────────────────────────────────────────────── */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !uploading && fileRef.current?.click()}
          className={`relative cursor-pointer p-8 rounded-2xl border-2 border-dashed transition-all text-center
            ${dragOver
              ? "border-green-500/50 bg-green-500/5"
              : "border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]"
            } ${uploading ? "cursor-wait opacity-70" : ""}`}>

          <input ref={fileRef} type="file" accept=".pdf" className="hidden"
            onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])} />

          {uploading ? (
            <div className="flex flex-col items-center gap-3">
              <div className="w-10 h-10 border-2 border-white/20 border-t-[#27AE60] rounded-full animate-spin" />
              <p className="text-sm text-white/50">{uploadMsg}</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center text-2xl"
                style={{ background: "linear-gradient(135deg, #1A5E20, #27AE60)" }}>📋</div>
              <div>
                <p className="text-sm font-medium text-white/70">Drop a contract PDF here, or click to browse</p>
                <p className="text-xs text-white/30 mt-1">AI extracts parties, dates & obligations · Saves to Supabase · Schedules Gmail reminders</p>
              </div>
            </div>
          )}
        </div>

        {/* Upload status message */}
        {uploadMsg && !uploading && (
          <div className={`text-sm px-4 py-3 rounded-xl border ${
            uploadMsg.startsWith("✅")
              ? "bg-green-500/10 border-green-500/30 text-green-300"
              : "bg-red-500/10 border-red-500/30 text-red-300"
          }`}>
            {uploadMsg}
          </div>
        )}

        {/* ── Natural language query ──────────────────────────────────────── */}
        <div className="p-6 rounded-2xl border border-white/8 bg-white/[0.02] space-y-4">
          <h3 className="text-sm font-medium text-white/60 uppercase tracking-wider">
            Query Your Contracts
          </h3>
          <div className="flex gap-3">
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleQuery()}
              placeholder="e.g. Which contracts expire this year? / List all auto-renewal agreements"
              className="flex-1 bg-white/[0.04] border border-white/8 rounded-xl px-4 py-3 text-sm text-white/80 placeholder-white/25 outline-none focus:border-[#2E75B6]/40 transition-colors"
            />
            <button
              onClick={handleQuery}
              disabled={querying || !query.trim()}
              className="px-5 py-3 rounded-xl text-sm font-medium transition-all disabled:opacity-30 hover:opacity-90"
              style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>
              {querying ? "Querying…" : "Ask"}
            </button>
          </div>

          {queryResp && (
            <div className="p-4 rounded-xl bg-white/[0.03] border border-white/6 text-sm text-white/70 leading-relaxed whitespace-pre-wrap">
              {queryResp}
            </div>
          )}

          {/* Quick query suggestions */}
          {!queryResp && (
            <div className="flex flex-wrap gap-2">
              {[
                "Which contracts expire within 30 days?",
                "List all auto-renewal contracts",
                "What are my obligations under service agreements?",
                "Which contracts have Indian law jurisdiction?",
              ].map(s => (
                <button key={s} onClick={() => { setQuery(s); }}
                  className="text-xs px-3 py-1.5 rounded-full border border-white/8 text-white/35 hover:border-white/20 hover:text-white/55 transition-all">
                  {s}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* ── Contracts table ────────────────────────────────────────────── */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-semibold text-white/70" style={{ fontFamily: "'Georgia', serif" }}>
              Stored Contracts
            </h2>
            <button onClick={fetchContracts}
              className="text-xs text-white/30 hover:text-white/60 transition-colors flex items-center gap-1.5">
              <span className={fetching ? "animate-spin" : ""}>↻</span>
              Refresh
            </button>
          </div>

          {fetching ? (
            <div className="text-center py-12 text-white/30 text-sm">Loading contracts…</div>
          ) : contracts.length === 0 ? (
            <div className="text-center py-16 rounded-2xl border border-white/6 bg-white/[0.02]">
              <p className="text-3xl mb-3">📭</p>
              <p className="text-white/40 text-sm">No contracts yet. Upload your first contract PDF above.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {contracts.map(c => {
                const days  = daysUntil(c.renewal_date);
                const badge = urgencyBadge(days);
                return (
                  <div key={c.id}
                    onClick={() => setSelected(selected?.id === c.id ? null : c)}
                    className="group cursor-pointer p-5 rounded-xl border border-white/8 bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04] transition-all">

                    <div className="flex items-start justify-between gap-4">
                      {/* Left: contract info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2.5 mb-2">
                          <span className="text-sm font-medium text-white/80 truncate"
                            style={{ fontFamily: "'Georgia', serif" }}>
                            {c.contract_type || "Contract"}
                          </span>
                          {c.auto_renewal && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-green-500/15 border border-green-500/25 text-green-400">
                              Auto-renew
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-white/40 truncate">
                          {c.party_a || "Unknown"} ↔ {c.party_b || "Unknown"}
                        </p>
                        <p className="text-[11px] text-white/25 mt-1 truncate">{c.filename}</p>
                      </div>

                      {/* Right: renewal status */}
                      <div className="text-right shrink-0 space-y-1.5">
                        <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-medium ${badge.bg}`}>
                          <span className={urgencyClass(days)}>●</span>
                          <span className={urgencyClass(days)}>{badge.label}</span>
                        </div>
                        <p className="text-[11px] text-white/30 block">
                          Renews: {formatDate(c.renewal_date)}
                        </p>
                      </div>
                    </div>

                    {/* Reminder status pills */}
                    <div className="flex gap-2 mt-3">
                      {([
                        { key: "reminder_sent_30", label: "30d reminder" },
                        { key: "reminder_sent_7",  label: "7d reminder"  },
                        { key: "reminder_sent_1",  label: "1d reminder"  },
                      ] as { key: keyof Contract; label: string }[]).map(({ key, label }) => (
                        <span key={key}
                          className={`text-[10px] px-2 py-0.5 rounded-full border ${
                            c[key]
                              ? "bg-green-500/10 border-green-500/20 text-green-400"
                              : "bg-white/[0.03] border-white/8 text-white/25"
                          }`}>
                          {c[key] ? "✓" : "○"} {label}
                        </span>
                      ))}
                    </div>

                    {/* Expanded detail panel */}
                    {selected?.id === c.id && (
                      <div className="mt-4 pt-4 border-t border-white/6 grid grid-cols-2 gap-x-6 gap-y-3">
                        {[
                          { label: "Start Date",      value: formatDate(c.start_date)          },
                          { label: "End Date",        value: formatDate(c.end_date)            },
                          { label: "Notice Period",   value: `${c.notice_period_days} days`    },
                          { label: "Governing Law",   value: c.governing_law || "—"            },
                        ].map(({ label, value }) => (
                          <div key={label}>
                            <p className="text-[10px] text-white/25 uppercase tracking-wide mb-0.5">{label}</p>
                            <p className="text-xs text-white/60">{value}</p>
                          </div>
                        ))}
                        {c.key_obligations && (
                          <div className="col-span-2">
                            <p className="text-[10px] text-white/25 uppercase tracking-wide mb-0.5">Key Obligations</p>
                            <p className="text-xs text-white/55 leading-relaxed">{c.key_obligations}</p>
                          </div>
                        )}
                        <div className="col-span-2">
                          <p className="text-[10px] text-white/25 uppercase tracking-wide mb-0.5">Contract ID</p>
                          <p className="text-[10px] font-mono text-white/25">{c.id}</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* ── How it works ──────────────────────────────────────────────── */}
        <div className="p-6 rounded-2xl border border-white/6 bg-white/[0.02]">
          <h3 className="text-xs uppercase tracking-widest text-white/25 mb-5 text-center">
            How Contract Intelligence Works
          </h3>
          <div className="grid grid-cols-5 gap-3">
            {[
              { icon: "📎", step: "Upload",    desc: "Drop any contract PDF"           },
              { icon: "🤖", step: "Extract",   desc: "Llama 3.3 reads and parses it"   },
              { icon: "🗄",  step: "Store",     desc: "Saved to Supabase Postgres"      },
              { icon: "📧", step: "Schedule",  desc: "Gmail reminders at 30 / 7 / 1d"  },
              { icon: "💬", step: "Query",     desc: "Ask questions in plain English"   },
            ].map(({ icon, step, desc }) => (
              <div key={step} className="text-center">
                <div className="text-2xl mb-2">{icon}</div>
                <div className="text-xs font-medium text-white/60 mb-1">{step}</div>
                <div className="text-[10px] text-white/30 leading-relaxed">{desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
