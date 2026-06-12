// app/page.tsx  —  LexRAG Landing Page
// Stack: Next.js 14 App Router · Tailwind CSS
// No external UI libraries needed — pure Tailwind + CSS variables

import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-[#0A0F1E] text-white overflow-x-hidden">

      {/* ── Ambient background grid ─────────────────────────────────────── */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(46,117,182,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(46,117,182,0.04) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      {/* ── Glow blobs ──────────────────────────────────────────────────── */}
      <div className="fixed top-[-200px] left-[-200px] w-[600px] h-[600px] rounded-full pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(46,117,182,0.12) 0%, transparent 70%)" }} />
      <div className="fixed bottom-[-200px] right-[-200px] w-[500px] h-[500px] rounded-full pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(31,56,100,0.18) 0%, transparent 70%)" }} />

      {/* ── Nav ─────────────────────────────────────────────────────────── */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6 border-b border-white/5">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>
            <span className="text-sm font-bold text-white">⚖</span>
          </div>
          <span className="font-semibold text-white tracking-tight"
            style={{ fontFamily: "'Georgia', serif", fontSize: "1.1rem" }}>
            LexRAG
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full border border-[#2E75B6]/40 text-[#2E75B6] ml-1">
            v2.0
          </span>
        </div>
        <div className="flex items-center gap-6">
          <Link href="/chat" className="text-sm text-white/60 hover:text-white transition-colors">
            Legal Chat
          </Link>
          <Link href="/contracts" className="text-sm text-white/60 hover:text-white transition-colors">
            Contracts
          </Link>
          <Link href="/chat"
            className="text-sm px-4 py-2 rounded-lg text-white font-medium transition-all hover:opacity-90"
            style={{ background: "linear-gradient(135deg, #2E75B6, #1F3864)" }}>
            Get Started
          </Link>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-5xl mx-auto px-8 pt-24 pb-20 text-center">

        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[#2E75B6]/30 bg-[#2E75B6]/10 text-[#7BB3E0] text-sm mb-8">
          <span className="w-2 h-2 rounded-full bg-[#2E75B6] animate-pulse" />
          Powered by InLegalBERT · LangGraph · Groq Llama 3.3
        </div>

        <h1 className="text-6xl font-bold leading-tight mb-6 tracking-tight"
          style={{ fontFamily: "'Georgia', serif" }}>
          Indian Legal Intelligence,{" "}
          <span style={{
            background: "linear-gradient(135deg, #2E75B6, #7BB3E0)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>
            Finally Accessible
          </span>
        </h1>

        <p className="text-lg text-white/50 max-w-2xl mx-auto leading-relaxed mb-10">
          Ask questions about the Indian Constitution, IPC, CrPC, and any legal document.
          Upload contracts for instant AI analysis and automated renewal reminders.
        </p>

        <div className="flex items-center justify-center gap-4">
          <Link href="/chat"
            className="px-8 py-3.5 rounded-xl text-white font-semibold text-sm transition-all hover:scale-105 hover:shadow-lg"
            style={{
              background: "linear-gradient(135deg, #2E75B6, #1F3864)",
              boxShadow: "0 0 30px rgba(46,117,182,0.3)",
            }}>
            Ask a Legal Question →
          </Link>
          <Link href="/contracts"
            className="px-8 py-3.5 rounded-xl font-semibold text-sm border border-white/10 text-white/70 hover:border-white/20 hover:text-white transition-all">
            Analyse a Contract
          </Link>
        </div>
      </section>

      {/* ── Stats bar ────────────────────────────────────────────────────── */}
      <section className="relative z-10 border-y border-white/5 bg-white/[0.02] py-8">
        <div className="max-w-4xl mx-auto px-8 grid grid-cols-4 gap-8 text-center">
          {[
            { value: "5,000+", label: "Legal Chunks Indexed" },
            { value: "7", label: "Agent Nodes" },
            { value: "4", label: "RAGAS Metrics" },
            { value: "Free", label: "Deployment Stack" },
          ].map(({ value, label }) => (
            <div key={label}>
              <div className="text-2xl font-bold text-white mb-1"
                style={{ fontFamily: "'Georgia', serif" }}>{value}</div>
              <div className="text-xs text-white/40 uppercase tracking-wider">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature cards ────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-5xl mx-auto px-8 py-20">
        <h2 className="text-2xl font-bold text-center mb-12 text-white/80"
          style={{ fontFamily: "'Georgia', serif" }}>
          Two Powerful Modules
        </h2>
        <div className="grid grid-cols-2 gap-6">

          {/* Legal Chat */}
          <Link href="/chat" className="group p-8 rounded-2xl border border-white/8 bg-white/[0.03] hover:border-[#2E75B6]/40 hover:bg-white/[0.05] transition-all">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-5"
              style={{ background: "linear-gradient(135deg, #1F3864, #2E75B6)" }}>
              <span className="text-2xl">⚖️</span>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3"
              style={{ fontFamily: "'Georgia', serif" }}>Legal Knowledge Chat</h3>
            <p className="text-sm text-white/50 leading-relaxed mb-4">
              Query the full Indian legal corpus — Constitution, IPC, CrPC, Evidence Act,
              and more. Upload your own documents for combined analysis.
            </p>
            <div className="flex flex-wrap gap-2">
              {["Constitution", "IPC", "CrPC", "Upload PDF", "Hybrid Search"].map(t => (
                <span key={t} className="text-xs px-2.5 py-1 rounded-full border border-white/10 text-white/40">
                  {t}
                </span>
              ))}
            </div>
            <div className="mt-5 text-sm text-[#2E75B6] group-hover:text-[#7BB3E0] transition-colors">
              Start asking →
            </div>
          </Link>

          {/* Contracts */}
          <Link href="/contracts" className="group p-8 rounded-2xl border border-white/8 bg-white/[0.03] hover:border-[#2E75B6]/40 hover:bg-white/[0.05] transition-all">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-5"
              style={{ background: "linear-gradient(135deg, #1A5E20, #27AE60)" }}>
              <span className="text-2xl">📋</span>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3"
              style={{ fontFamily: "'Georgia', serif" }}>Contract Intelligence</h3>
            <p className="text-sm text-white/50 leading-relaxed mb-4">
              Upload any contract PDF. The AI extracts parties, dates, and obligations,
              stores them in your database, and reminds you before renewal deadlines.
            </p>
            <div className="flex flex-wrap gap-2">
              {["Auto-Extract", "Renewal Alerts", "Gmail SMTP", "Supabase", "2-min Test"].map(t => (
                <span key={t} className="text-xs px-2.5 py-1 rounded-full border border-white/10 text-white/40">
                  {t}
                </span>
              ))}
            </div>
            <div className="mt-5 text-sm text-[#27AE60] group-hover:text-[#58D68D] transition-colors">
              Analyse contracts →
            </div>
          </Link>
        </div>
      </section>

      {/* ── Tech stack ───────────────────────────────────────────────────── */}
      <section className="relative z-10 max-w-5xl mx-auto px-8 pb-20">
        <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-8">
          <h3 className="text-sm uppercase tracking-widest text-white/30 mb-6 text-center">
            Architecture
          </h3>
          <div className="grid grid-cols-3 gap-6 text-sm">
            {[
              { layer: "Embeddings", tech: "InLegalBERT + BGE-Base-en" },
              { layer: "Vector DB", tech: "ChromaDB → Pinecone" },
              { layer: "LLM", tech: "Groq Llama 3.3 70B" },
              { layer: "Agent Framework", tech: "LangGraph 7-node DAG" },
              { layer: "Backend", tech: "FastAPI + Render" },
              { layer: "Frontend", tech: "Next.js 14 + Vercel" },
              { layer: "Database", tech: "Supabase (Postgres)" },
              { layer: "Evaluation", tech: "RAGAS 0.2.x" },
              { layer: "Email", tech: "Gmail SMTP (smtplib)" },
            ].map(({ layer, tech }) => (
              <div key={layer} className="flex gap-3 items-start">
                <div className="w-1.5 h-1.5 rounded-full bg-[#2E75B6] mt-1.5 shrink-0" />
                <div>
                  <div className="text-white/30 text-xs uppercase tracking-wide">{layer}</div>
                  <div className="text-white/70 mt-0.5">{tech}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <footer className="relative z-10 border-t border-white/5 px-8 py-6 text-center text-xs text-white/20">
        LexRAG v2.0 · Indian Legal AI · Built with Next.js, FastAPI, LangGraph · Deployed on Vercel + Render
      </footer>
    </main>
  );
}
