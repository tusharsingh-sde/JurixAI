import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp, Check, FileText, Mic,
  LoaderCircle, LogOut, Menu, MessageCircle, Paperclip, Plus, Scale,
  ShieldCheck, Sparkles, Trash2, X, Gavel,
} from 'lucide-react'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${API_URL}${path}`, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    let errorMsg = data.detail || data.message || `Request failed (${response.status})`
    if (Array.isArray(errorMsg)) {
      errorMsg = errorMsg.map((err) => err.msg || JSON.stringify(err)).join(', ')
    } else if (typeof errorMsg === 'object') {
      errorMsg = JSON.stringify(errorMsg)
    }
    throw new Error(errorMsg)
  }
  return data
}

// ── Suggestions shown on the empty-state welcome screen ────────────────
const SUGGESTIONS = [
  { icon: <Gavel size={17} />, text: 'What are my rights if my landlord is evicting me without notice?' },
  { icon: <FileText size={17} />, text: 'Explain FIR filing procedure under CrPC in simple terms.' },
  { icon: <ShieldCheck size={17} />, text: 'Can my employer deduct salary without written notice?' },
  { icon: <Scale size={17} />, text: 'What is the punishment for cyber fraud under IT Act 2000?' },
]

// ── Render an assistant message with markdown-lite formatting ───────────
function formatContent(text) {
  // Bold **text** and *text*
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('*') && part.endsWith('*'))
      return <em key={i}>{part.slice(1, -1)}</em>
    return part
  })
}

function Message({ message }) {
  const isUser = message.role === 'user'
  if (isUser) {
    return (
      <div className="msg-row msg-user">
        <div className="msg-bubble msg-bubble-user">{message.content}</div>
      </div>
    )
  }
  // Split assistant reply on newlines for nicer rendering
  const lines = message.content.split('\n')
  return (
    <div className="msg-row msg-assistant">
      <div className="ai-avatar"><Scale size={15} /></div>
      <div className="msg-content">
        <div className="msg-name">JurixAI <span>· Legal assistant</span></div>
        <div className="msg-bubble msg-bubble-assistant">
          {lines.map((line, i) => (
            <p key={i} className={line.trim() === '' ? 'msg-spacer' : undefined}>
              {formatContent(line)}
            </p>
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Main App ─────────────────────────────────────────────────────────────
export default function App() {
  const [token, setToken]               = useState(() => localStorage.getItem('jurixai_token'))
  const [email, setEmail]               = useState(() => localStorage.getItem('jurixai_email') || '')
  const [sessions, setSessions]         = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages]         = useState([])
  const [draft, setDraft]               = useState('')
  const [busy, setBusy]                 = useState(false)
  const [uploading, setUploading]       = useState(false)
  const [pendingFile, setPendingFile]   = useState(null)
  const [listening, setListening]       = useState(false)
  const [sidebarOpen, setSidebarOpen]   = useState(false)
  const [error, setError]               = useState('')
  const [notice, setNotice]             = useState('')

  // Auth modal state
  const [authOpen, setAuthOpen]         = useState(false)
  const [authMode, setAuthMode]         = useState('login')  // 'login' | 'signup'
  const [authEmail, setAuthEmail]       = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authBusy, setAuthBusy]         = useState(false)
  const [authError, setAuthError]       = useState('')
  const [authNotice, setAuthNotice]     = useState('')
  const [pendingPrompt, setPendingPrompt] = useState('')

  const endRef   = useRef(null)
  const inputRef = useRef(null)
  const fileRef  = useRef(null)

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])
  useEffect(() => { if (token) loadSessions() }, [token])
  useEffect(() => {
    if (token && pendingPrompt) {
      const p = pendingPrompt
      setPendingPrompt('')
      sendMessage(p)
    }
  }, [token])

  // ── Sessions ───────────────────────────────────────────────────────────
  async function loadSessions() {
    try {
      const data = await request('/sessions', { token })
      setSessions(data.sessions || [])
    } catch (e) { setError(e.message) }
  }

  async function openSession(session) {
    setActiveSession(session)
    setSidebarOpen(false)
    setError('')
    try {
      const data = await request(`/sessions/${session.id}/messages`, { token })
      setMessages((data.messages || []).map((m) => ({
        id: m.id,
        role: m.role?.toLowerCase() === 'user' ? 'user' : 'assistant',
        content: m.content,
      })))
    } catch (e) { setError(e.message); setMessages([]) }
  }

  function newConversation() {
    setActiveSession(null)
    setMessages([])
    setDraft('')
    setError('')
    setNotice('')
    setSidebarOpen(false)
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  async function removeSession(e, session) {
    e.stopPropagation()
    try {
      await request(`/sessions/${session.id}`, { method: 'DELETE', token })
      setSessions((cur) => cur.filter((s) => s.id !== session.id))
      if (activeSession?.id === session.id) newConversation()
    } catch (e) { setError(e.message) }
  }

  // ── Auth ───────────────────────────────────────────────────────────────
  function openAuth(mode = 'login') {
    setAuthMode(mode)
    setAuthError('')
    setAuthNotice('')
    setAuthOpen(true)
  }

  async function authenticate(event) {
    event.preventDefault()
    setAuthError('')
    setAuthBusy(true)
    try {
      const data = await request(authMode === 'login' ? '/login' : '/signup', {
        method: 'POST',
        body: JSON.stringify({ email: authEmail.trim(), password: authPassword }),
      })
      if (authMode === 'signup') {
        setAuthNotice('Account created! Sign in to continue.')
        setAuthMode('login')
      } else {
        localStorage.setItem('jurixai_token', data.access_token)
        localStorage.setItem('jurixai_email', data.email || authEmail.trim())
        setEmail(data.email || authEmail.trim())
        setToken(data.access_token)
        setAuthOpen(false)
        setAuthPassword('')
      }
    } catch (e) { setAuthError(e.message) }
    finally { setAuthBusy(false) }
  }

  async function signOut() {
    try { await request('/logout', { method: 'POST', token }) } catch { /* ignore */ }
    localStorage.removeItem('jurixai_token')
    localStorage.removeItem('jurixai_email')
    setToken(null); setEmail(''); setSessions([]); setActiveSession(null); setMessages([])
  }

  // ── Chat ───────────────────────────────────────────────────────────────
  async function sendMessage(text = draft, documentText = null) {
    const content = (text || '').trim()
    if (!content || busy) return
    if (!token) {
      setDraft(content)
      setPendingPrompt(content)
      openAuth('login')
      return
    }
    setDraft(''); setError(''); setNotice('')
    const userMsg = { id: crypto.randomUUID(), role: 'user', content }
    setMessages((cur) => [...cur, userMsg])
    setBusy(true)
    try {
      const payload = { session_id: activeSession?.id || null, message: content }
      if (documentText) payload.document_text = documentText
      const data = await request('/chat', { method: 'POST', token, body: JSON.stringify(payload) })
      setMessages((cur) => [...cur, { id: crypto.randomUUID(), role: 'assistant', content: data.reply }])
      if (data.session_id && data.session_id !== activeSession?.id) {
        setActiveSession({ id: data.session_id, title: content.length > 50 ? `${content.slice(0, 50)}…` : content })
      }
      await loadSessions()
    } catch (e) {
      setError(e.message)
      setMessages((cur) => cur.filter((m) => m.id !== userMsg.id))
      setDraft(content)
    } finally { setBusy(false) }
  }

  // ── File upload ────────────────────────────────────────────────────────
  async function analyzeFile(file) {
    if (!file) return
    if (!token) { openAuth('login'); return }
    setPendingFile(file); setUploading(true); setError(''); setNotice('')
    const formData = new FormData(); formData.append('file', file)
    try {
      const data = await request('/upload', { method: 'POST', token, body: formData })
      if (!data.extracted_text?.trim()) throw new Error('No readable text found. Try a clearer scan.')
      setPendingFile(null)
      await sendMessage(
        `Please review this uploaded ${file.name} and explain the important details, risks, deadlines, and what I should do next.`,
        data.extracted_text
      )
    } catch (e) { setError(e.message); setPendingFile(null) }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = '' }
  }

  function handleSubmit(e) { e.preventDefault(); sendMessage() }

  // ── Voice ──────────────────────────────────────────────────────────────
  function startVoice() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SR) { setNotice('Voice input not available in this browser.'); return }
    const r = new SR(); r.lang = 'en-IN'; r.interimResults = false
    r.onstart = () => setListening(true)
    r.onend = () => setListening(false)
    r.onerror = () => { setListening(false); setError('Microphone permission denied.') }
    r.onresult = (e) => setDraft((cur) => `${cur}${cur ? ' ' : ''}${e.results[0][0].transcript}`)
    r.start()
  }

  // ── Auto-resize textarea ───────────────────────────────────────────────
  function autoResize(e) {
    e.target.style.height = 'auto'
    e.target.style.height = `${Math.min(e.target.scrollHeight, 180)}px`
  }

  // ── Show auth modal when not logged in ────────────────────────────────
  const showAuthGate = !token

  return (
    <div className="shell">
      {/* ── Auth Modal ──────────────────────────────────────────────── */}
      {authOpen && (
        <div className="modal-backdrop" onClick={() => setAuthOpen(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setAuthOpen(false)} aria-label="Close"><X size={18} /></button>
            <div className="modal-brand"><Scale size={20} /><span>JurixAI</span></div>
            <h2>{authMode === 'login' ? 'Welcome back' : 'Create account'}</h2>
            <p className="modal-sub">
              {authMode === 'login'
                ? 'Sign in to access your legal conversations.'
                : 'Start your journey with AI-powered Indian law guidance.'}
            </p>
            {authError && <div className="modal-error">{authError}</div>}
            {authNotice && <div className="modal-notice"><Check size={14} />{authNotice}</div>}
            <form onSubmit={authenticate} className="modal-form">
              <label>
                Email
                <input type="email" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)}
                  placeholder="you@example.com" autoComplete="email" required />
              </label>
              <label>
                Password
                <input type="password" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)}
                  placeholder={authMode === 'signup' ? 'Min 8 characters' : 'Your password'}
                  autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                  minLength={authMode === 'signup' ? 8 : 1} required />
              </label>
              <button className="modal-submit" disabled={authBusy}>
                {authBusy ? <LoaderCircle className="spin" size={17} /> : (authMode === 'login' ? 'Sign in' : 'Create account')}
              </button>
            </form>
            <p className="modal-switch">
              {authMode === 'login' ? "Don't have an account?" : 'Already have an account?'}
              {' '}
              <button onClick={() => { setAuthMode(authMode === 'login' ? 'signup' : 'login'); setAuthError(''); setAuthNotice('') }}>
                {authMode === 'login' ? 'Sign up' : 'Sign in'}
              </button>
            </p>
            <div className="modal-assurance"><ShieldCheck size={13} />Your data is private and encrypted.</div>
          </div>
        </div>
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      {sidebarOpen && <div className="sidebar-scrim" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sb-header">
          <div className="sb-brand"><Scale size={17} /><span>Jurix<b>AI</b></span></div>
          <button className="sb-close icon-btn" onClick={() => setSidebarOpen(false)} aria-label="Close"><X size={17} /></button>
        </div>

        <button className="new-chat-btn" onClick={newConversation}>
          <Plus size={16} /> New conversation
        </button>

        <div className="sb-section-label">Conversations</div>
        <div className="session-list">
          {sessions.length === 0 && <p className="sessions-empty">No conversations yet.</p>}
          {sessions.map((s) => (
            <button key={s.id}
              className={`session-item ${activeSession?.id === s.id ? 'active' : ''}`}
              onClick={() => openSession(s)} title={s.title}>
              <MessageCircle size={14} />
              <span className="session-title">{s.title || 'New conversation'}</span>
              <span className="session-del" onClick={(e) => removeSession(e, s)} aria-label="Delete">
                <Trash2 size={13} />
              </span>
            </button>
          ))}
        </div>

        <div className="sb-footer">
          {token ? (
            <div className="sb-profile">
              <div className="sb-avatar">{email.charAt(0).toUpperCase()}</div>
              <div className="sb-profile-info">
                <strong>{email.split('@')[0]}</strong>
                <small>Free plan</small>
              </div>
              <button className="icon-btn" onClick={signOut} title="Sign out"><LogOut size={16} /></button>
            </div>
          ) : (
            <div className="sb-auth-cta">
              <button className="sb-signin-btn" onClick={() => openAuth('login')}>Sign in</button>
              <button className="sb-signup-btn" onClick={() => openAuth('signup')}>Create account</button>
            </div>
          )}
          <div className="sb-privacy"><ShieldCheck size={12} />Conversations are private &amp; encrypted.</div>
        </div>
      </aside>

      {/* ── Main panel ──────────────────────────────────────────────── */}
      <div className="main">
        {/* Topbar */}
        <header className="topbar">
          <button className="icon-btn menu-btn" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="topbar-brand"><Scale size={18} /><span>JurixAI</span></div>
          <div className="topbar-right">
            {showAuthGate ? (
              <>
                <button className="tb-ghost" onClick={() => openAuth('login')}>Sign in</button>
                <button className="tb-primary" onClick={() => openAuth('signup')}>Get started</button>
              </>
            ) : (
              <>
                <span className="tb-secure"><ShieldCheck size={13} /> Private session</span>
                <button className="tb-ghost" onClick={newConversation}><Plus size={14} /> New chat</button>
              </>
            )}
          </div>
        </header>

        {/* Conversation area */}
        <section className="conversation" aria-live="polite">
          {messages.length === 0 ? (
            /* ── Welcome / empty state ─────────────────────────────── */
            <div className="welcome">
              <div className="welcome-logo"><Scale size={32} /></div>
              <h1>Your Legal Saathi</h1>
              <p className="welcome-sub">
                Ask anything about Indian law — contracts, tenancy, FIRs, consumer rights, and more.
              </p>

              {showAuthGate && (
                <div className="welcome-auth">
                  <button className="wa-primary" onClick={() => openAuth('signup')}>
                    Create free account
                  </button>
                  <button className="wa-ghost" onClick={() => openAuth('login')}>
                    Sign in
                  </button>
                </div>
              )}

              <div className="suggestion-grid">
                {SUGGESTIONS.map((s, i) => (
                  <button key={i} className="suggestion-card"
                    onClick={() => sendMessage(s.text)}>
                    <div className="sug-icon">{s.icon}</div>
                    <span>{s.text}</span>
                  </button>
                ))}
              </div>

              <div className="welcome-footer">
                <Sparkles size={12} />
                JurixAI may make mistakes. Consult a lawyer for important matters.
              </div>
            </div>
          ) : (
            /* ── Message list ───────────────────────────────────────── */
            <div className="message-list">
              {messages.map((m) => <Message key={m.id} message={m} />)}

              {busy && (
                <div className="msg-row msg-assistant">
                  <div className="ai-avatar"><Scale size={15} /></div>
                  <div className="msg-content">
                    <div className="msg-name">JurixAI <span>· Thinking…</span></div>
                    <div className="typing-indicator">
                      <span /><span /><span />
                    </div>
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>
          )}
        </section>

        {/* ── Composer ──────────────────────────────────────────────── */}
        <div className="composer-zone">
          {error && (
            <div className="chat-alert">
              <span>{error}</span>
              <button onClick={() => setError('')}><X size={14} /></button>
            </div>
          )}
          {notice && (
            <div className="chat-notice">
              <Check size={14} /><span>{notice}</span>
              <button onClick={() => setNotice('')}><X size={14} /></button>
            </div>
          )}
          {pendingFile && (
            <div className="file-chip">
              <FileText size={14} />
              <span>{pendingFile.name}</span>
              {uploading && <LoaderCircle className="spin" size={13} />}
              <button onClick={() => { setPendingFile(null); setUploading(false) }}><X size={13} /></button>
            </div>
          )}

          <form className="composer" onSubmit={handleSubmit}>
            {/* Hidden file input */}
            <input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg" hidden
              onChange={(e) => analyzeFile(e.target.files?.[0])} />

            <button type="button" className="tool-btn attach-btn"
              onClick={() => token ? fileRef.current?.click() : openAuth('login')}
              disabled={busy || uploading} aria-label="Attach document">
              <Paperclip size={20} />
            </button>

            <textarea
              ref={inputRef}
              value={draft}
              onChange={(e) => { setDraft(e.target.value); autoResize(e) }}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(e) } }}
              placeholder={token ? 'Ask anything about Indian law…' : 'Sign in to start chatting…'}
              rows={1}
              aria-label="Your message"
            />

            <button type="button"
              className={`tool-btn mic-btn ${listening ? 'listening' : ''}`}
              onClick={startVoice} aria-label="Voice input">
              <Mic size={20} />
            </button>

            <button className="send-btn" type="submit"
              disabled={!draft.trim() || busy || uploading} aria-label="Send">
              {busy ? <LoaderCircle className="spin" size={18} /> : <ArrowUp size={18} />}
            </button>
          </form>

          <p className="composer-disclaimer">
            <Sparkles size={11} /> JurixAI can make mistakes. For important matters, consult a qualified lawyer.
          </p>
        </div>
      </div>
    </div>
  )
}
