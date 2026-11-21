import { useEffect, useRef, useState } from 'react'
import {
  Check, FileText, LoaderCircle, LogOut, Menu,
  MessageCircle, Plus, Scale, ShieldCheck, Sparkles, Trash2, X,
} from 'lucide-react'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

async function request(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }
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

function App() {
  const [token, setToken] = useState(() => localStorage.getItem('jurixai_token'))
  const [authOpen, setAuthOpen] = useState(false)
  const [pendingPrompt, setPendingPrompt] = useState('')
  const [pendingUpload, setPendingUpload] = useState(false)
  const [email, setEmail] = useState(() => localStorage.getItem('jurixai_email') || '')
  const [authMode, setAuthMode] = useState('login')
  const [sessions, setSessions] = useState([])
  const [activeSession, setActiveSession] = useState(null)
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [authBusy, setAuthBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [authEmail, setAuthEmail] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [pendingFile, setPendingFile] = useState(null)
  const [listening, setListening] = useState(false)
  const [uploading, setUploading] = useState(false)

  const endRef = useRef(null)
  const inputRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  useEffect(() => {
    if (token) loadSessions()
  }, [token])

  useEffect(() => {
    if (token && pendingPrompt) {
      const prompt = pendingPrompt
      setPendingPrompt('')
      sendMessage(prompt)
    } else if (token && pendingUpload) {
      setPendingUpload(false)
      setTimeout(() => fileRef.current?.click(), 120)
    }
  }, [token])

  useEffect(() => {
    if (!token) return
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        newConversation()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [token])

  async function loadSessions() {
    try {
      const data = await request('/sessions', { token })
      setSessions(data.sessions || [])
    } catch (e) {
      setError(e.message)
    }
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
    } catch (e) {
      setError(e.message)
      setMessages([])
    }
  }

  function newConversation() {
    setActiveSession(null)
    setMessages([])
    setDraft('')
    setError('')
    setNotice('')
    setSidebarOpen(false)
    setTimeout(() => inputRef.current?.focus(), 60)
  }

  async function authenticate(event) {
    event.preventDefault()
    setError('')
    setAuthBusy(true)
    try {
      const data = await request(authMode === 'login' ? '/login' : '/signup', {
        method: 'POST',
        body: JSON.stringify({ email: authEmail.trim(), password: authPassword }),
      })
      if (authMode === 'signup') {
        setNotice('Your account is ready. Sign in to start a conversation.')
        setAuthMode('login')
      } else {
        localStorage.setItem('jurixai_token', data.access_token)
        localStorage.setItem('jurixai_email', data.email || authEmail.trim())
        setEmail(data.email || authEmail.trim())
        setToken(data.access_token)
        setAuthOpen(false)
        setAuthPassword('')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setAuthBusy(false)
    }
  }

  async function signOut() {
    try { await request('/logout', { method: 'POST', token }) } catch { /* ignore */ }
    localStorage.removeItem('jurixai_token')
    localStorage.removeItem('jurixai_email')
    setToken(null)
    setEmail('')
    setSessions([])
    setActiveSession(null)
    setMessages([])
  }

  async function removeSession(event, session) {
    event.stopPropagation()
    try {
      await request(`/sessions/${session.id}`, { method: 'DELETE', token })
      setSessions((current) => current.filter((item) => item.id !== session.id))
      if (activeSession?.id === session.id) newConversation()
    } catch (e) {
      setError(e.message)
    }
  }

  async function sendMessage(text = draft, documentText = null) {
    const content = text.trim()
    if (!content || busy) return
    if (!token) {
      setDraft(content)
      setPendingPrompt(content)
      setAuthOpen(true)
      return
    }
    setDraft('')
    setError('')
    setNotice('')
    const userMessage = { id: crypto.randomUUID(), role: 'user', content }
    setMessages((current) => [...current, userMessage])
    setBusy(true)
    try {
      const payload = { session_id: activeSession?.id || null, message: content }
      if (documentText) {
        payload.document_text = documentText
      } else if (content.includes('\n\nDOCUMENT TEXT:\n')) {
        const [msgPart, docPart] = content.split('\n\nDOCUMENT TEXT:\n')
        payload.message = msgPart
        payload.document_text = docPart
      }
      const data = await request('/chat', {
        method: 'POST',
        token,
        body: JSON.stringify(payload),
      })
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'assistant', content: data.reply }])
      if (data.session_id && data.session_id !== activeSession?.id) {
        setActiveSession({
          id: data.session_id,
          title: content.length > 50 ? `${content.slice(0, 50)}...` : content,
        })
      }
      await loadSessions()
    } catch (e) {
      setError(e.message)
      setMessages((current) => current.filter((m) => m.id !== userMessage.id))
      setDraft(content)
    } finally {
      setBusy(false)
    }
  }

  async function analyzeFile(file) {
    if (!file) return
    setPendingFile(file)
    setUploading(true)
    setError('')
    setNotice('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const data = await request('/upload', { method: 'POST', token, body: formData })
      if (!data.extracted_text?.trim()) {
        throw new Error('No readable text was found. Try a clearer scan or image.')
      }
      setPendingFile(null)
      await sendMessage(
        `Please review this uploaded ${file.name} and explain the important details, risks, deadlines, and what I should do next.`,
        data.extracted_text
      )
    } catch (e) {
      setError(e.message)
      setPendingFile(null)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  function startVoiceInput(onTranscript) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      setNotice('Voice input is not available in this browser. You can type your question instead.')
      return
    }
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-IN'
    recognition.interimResults = false
    recognition.onstart = () => setListening(true)
    recognition.onend = () => setListening(false)
    recognition.onerror = () => {
      setListening(false)
      setError('Microphone input could not start. Check your browser microphone permission.')
    }
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      onTranscript(transcript)
    }
    recognition.start()
  }

  if (!token && !authOpen) {
    return (
      <HeroLanding
        onAsk={(prompt) => {
          setDraft(prompt)
          setPendingPrompt(prompt)
          setAuthOpen(true)
        }}
        onUpload={() => {
          setPendingUpload(true)
          setAuthOpen(true)
        }}
        onSignIn={() => setAuthOpen(true)}
        listening={listening}
        startVoiceInput={startVoiceInput}
      />
    )
  }

  if (!token) {
    return (
      <AuthScreen
        authMode={authMode}
        setAuthMode={setAuthMode}
        authEmail={authEmail}
        setAuthEmail={setAuthEmail}
        authPassword={authPassword}
        setAuthPassword={setAuthPassword}
        authenticate={authenticate}
        authBusy={authBusy}
        error={error}
        notice={notice}
        onBack={() => { setAuthOpen(false); setError(''); setNotice('') }}
      />
    )
  }

  const inChat = messages.length > 0

  return (
    <div className="cobalt-shell">
      {sidebarOpen && (
        <button className="sidebar-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-head">
          <div className="sidebar-brand">
            <span className="sidebar-brand-icon"><Scale size={18} /></span>
            <span>Jurix<strong>AI</strong></span>
          </div>
          <button className="icon-btn" onClick={() => setSidebarOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>

        <button className="new-chat-btn" onClick={newConversation}>
          <Plus size={16} />
          <span>New conversation</span>
          <kbd>⌘K</kbd>
        </button>

        <div className="sidebar-section-label">
          <span>YOUR CONVERSATIONS</span>
          <span className="count-pill">{sessions.length}</span>
        </div>

        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={`session-item ${activeSession?.id === session.id ? 'selected' : ''}`}
              onClick={() => openSession(session)}
              title={session.title}
            >
              <MessageCircle size={15} />
              <span className="session-title">{session.title || 'New conversation'}</span>
              <span
                className="session-delete"
                onClick={(event) => removeSession(event, session)}
                role="button"
                aria-label="Delete conversation"
              >
                <Trash2 size={14} />
              </span>
            </button>
          ))}
          {sessions.length === 0 && (
            <p className="empty-sessions">Your legal chats will appear here.</p>
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="privacy-note">
            <ShieldCheck size={15} />
            <span>Private & securely saved</span>
          </div>
          <div className="profile-row">
            <div className="avatar">{email.charAt(0).toUpperCase() || 'J'}</div>
            <div className="profile-info">
              <strong>{email.split('@')[0]}</strong>
              <span>Free account</span>
            </div>
            <button className="icon-btn" onClick={signOut} aria-label="Sign out" title="Sign out">
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      <main className={`workspace ${inChat ? 'workspace-chat' : 'workspace-hero'}`}>
        <header className="topbar">
          <button className="icon-btn topbar-menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          {inChat && (
            <button className="topbar-brand" onClick={newConversation}>
              <Scale size={18} />
              <span>JurixAI</span>
            </button>
          )}
          <div className="topbar-right">
            <span className="secure-pill">
              <ShieldCheck size={14} /> Private session
            </span>
            <button className="topbar-new-btn" onClick={newConversation}>
              <Plus size={15} /> New chat
            </button>
          </div>
        </header>

        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
          hidden
          onChange={(event) => analyzeFile(event.target.files?.[0])}
        />

        {!inChat ? (
          <div className="hero-stage">
            <div className="hero-content">
              <JurixLogo />
              <h1 className="hero-headline">
                ELITE INDIAN<br />LEGAL<br />INTELLIGENCE
              </h1>
            </div>

            <img
              className="justice-statue"
              src="/lady-justice.png"
              alt="Blue-lit Lady Justice holding balanced scales"
            />

            <div className="hero-composer-wrap">
              <StatusBanners
                error={error}
                notice={notice}
                pendingFile={pendingFile}
                uploading={uploading}
                onClearError={() => setError('')}
                onClearNotice={() => setNotice('')}
                onClearFile={() => { setPendingFile(null); setUploading(false) }}
              />
              <GlassPillForm
                inputRef={inputRef}
                value={draft}
                onChange={setDraft}
                onSubmit={() => sendMessage()}
                onAttach={() => fileRef.current?.click()}
                onVoice={() => startVoiceInput((t) => setDraft((c) => `${c}${c ? ' ' : ''}${t}`))}
                listening={listening}
                busy={busy}
                uploading={uploading}
                placeholder="Ask anything about Indian law..."
              />
            </div>
          </div>
        ) : (
          <div className="chat-stage">
            <img className="chat-watermark" src="/lady-justice.png" alt="" aria-hidden="true" />
            <section className="chat-scroll">
              <div className="message-list">
                {messages.map((message) => (
                  <Message key={message.id} message={message} />
                ))}
                {busy && (
                  <div className="typing-row">
                    <div className="assistant-avatar"><Scale size={15} /></div>
                    <div className="typing-dots"><span /><span /><span /></div>
                    <small>Analyzing Indian law...</small>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            </section>

            <footer className="chat-footer">
              <StatusBanners
                error={error}
                notice={notice}
                pendingFile={pendingFile}
                uploading={uploading}
                onClearError={() => setError('')}
                onClearNotice={() => setNotice('')}
                onClearFile={() => { setPendingFile(null); setUploading(false) }}
              />
              <GlassPillForm
                inputRef={inputRef}
                value={draft}
                onChange={setDraft}
                onSubmit={() => sendMessage()}
                onAttach={() => fileRef.current?.click()}
                onVoice={() => startVoiceInput((t) => setDraft((c) => `${c}${c ? ' ' : ''}${t}`))}
                listening={listening}
                busy={busy}
                uploading={uploading}
                placeholder="Ask a follow-up question..."
                compact
              />
              <div className="disclaimer">
                <Sparkles size={11} /> JurixAI can make mistakes. For critical legal matters, consult a qualified advocate.
              </div>
            </footer>
          </div>
        )}
      </main>
    </div>
  )
}

function StatusBanners({ error, notice, pendingFile, uploading, onClearError, onClearNotice, onClearFile }) {
  return (
    <>
      {error && (
        <div className="status-banner status-error">
          <span>{error}</span>
          <button type="button" onClick={onClearError} aria-label="Dismiss"><X size={14} /></button>
        </div>
      )}
      {notice && (
        <div className="status-banner status-notice">
          <Check size={14} />
          <span>{notice}</span>
          <button type="button" onClick={onClearNotice} aria-label="Dismiss"><X size={14} /></button>
        </div>
      )}
      {pendingFile && (
        <div className="file-chip">
          <FileText size={14} />
          <span>{pendingFile.name}</span>
          {uploading && <LoaderCircle className="spin" size={14} />}
          <button type="button" onClick={onClearFile} aria-label="Remove file"><X size={14} /></button>
        </div>
      )}
    </>
  )
}

function GlassPillForm({
  inputRef, value, onChange, onSubmit, onAttach, onVoice,
  listening, busy, uploading, placeholder, compact = false,
}) {
  return (
    <form
      className={`glass-pill ${compact ? 'glass-pill-compact' : ''}`}
      onSubmit={(e) => { e.preventDefault(); onSubmit() }}
    >
      <button
        type="button"
        className="pill-icon-btn"
        onClick={onAttach}
        disabled={busy || uploading}
        aria-label="Attach document"
        title="Upload PDF or image"
      >
        <UprightPaperclipIcon />
      </button>

      <input
        ref={inputRef}
        type="text"
        className="pill-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        aria-label="Ask a legal question"
      />

      <div className="pill-actions">
        <button
          type="button"
          className={`pill-icon-btn mic-btn ${listening ? 'listening' : ''}`}
          onClick={onVoice}
          aria-label="Voice input"
          title="Voice input"
        >
          <FilledMicIcon />
        </button>

        <button
          type="submit"
          className="pill-send-btn"
          disabled={busy || uploading}
          aria-label="Send message"
        >
          {busy ? <LoaderCircle className="spin" size={19} /> : <RightArrowIcon />}
        </button>
      </div>
    </form>
  )
}

function HeroLanding({ onAsk, onUpload, onSignIn, listening, startVoiceInput }) {
  const [query, setQuery] = useState('')
  return (
    <main className="cobalt-shell landing-shell">
      <button className="landing-signin-btn" onClick={onSignIn}>Sign in</button>
      <div className="hero-stage">
        <div className="hero-content">
          <JurixLogo />
          <h1 className="hero-headline">
            ELITE INDIAN<br />LEGAL<br />INTELLIGENCE
          </h1>
        </div>

        <img
          className="justice-statue"
          src="/lady-justice.png"
          alt="Blue-lit Lady Justice holding balanced scales"
        />

        <div className="hero-composer-wrap">
          <GlassPillForm
            value={query}
            onChange={setQuery}
            onSubmit={() => onAsk(query.trim() || 'I have a legal question')}
            onAttach={onUpload}
            onVoice={() => startVoiceInput((t) => setQuery((c) => `${c}${c ? ' ' : ''}${t}`))}
            listening={listening}
            busy={false}
            uploading={false}
            placeholder="Ask a legal question..."
          />
        </div>
      </div>
    </main>
  )
}

function AuthScreen({
  authMode, setAuthMode, authEmail, setAuthEmail,
  authPassword, setAuthPassword, authenticate, authBusy, error, notice, onBack,
}) {
  return (
    <main className="cobalt-shell landing-shell">
      <div className="hero-stage auth-stage">
        <img className="justice-statue auth-statue" src="/lady-justice.png" alt="" aria-hidden="true" />

        <div className="hero-content auth-bg-brand">
          <JurixLogo />
        </div>

        <button className="auth-back-btn" onClick={onBack}>
          <X size={15} /> Back
        </button>

        <div className="auth-card">
          <div className="auth-kicker">SECURE ACCESS · JURIXAI</div>
          <h2>{authMode === 'login' ? 'Welcome back.' : 'Create your account.'}</h2>
          <p className="auth-desc">
            {authMode === 'login'
              ? 'Sign in to continue your private legal conversations.'
              : 'Get law-grounded Indian legal assistance in seconds.'}
          </p>

          {error && <div className="status-banner status-error"><span>{error}</span></div>}
          {notice && (
            <div className="status-banner status-notice">
              <Check size={14} /><span>{notice}</span>
            </div>
          )}

          <form onSubmit={authenticate} className="auth-form">
            <label>
              Email address
              <input
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={authEmail}
                onChange={(e) => setAuthEmail(e.target.value)}
                required
              />
            </label>
            <label>
              Password
              <input
                type="password"
                autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                placeholder={authMode === 'login' ? 'Your password' : 'At least 8 characters'}
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                minLength={authMode === 'signup' ? 8 : 1}
                required
              />
            </label>
            <button className="auth-submit-btn" type="submit" disabled={authBusy}>
              {authBusy ? <LoaderCircle className="spin" size={18} /> : (
                <>
                  <span>{authMode === 'login' ? 'Sign in' : 'Create account'}</span>
                  <RightArrowIcon />
                </>
              )}
            </button>
          </form>

          <p className="auth-switch">
            {authMode === 'login' ? 'New to JurixAI?' : 'Already have an account?'}{' '}
            <button type="button" onClick={() => setAuthMode(authMode === 'login' ? 'signup' : 'login')}>
              {authMode === 'login' ? 'Create an account' : 'Sign in'}
            </button>
          </p>

          <div className="auth-assurance">
            <ShieldCheck size={14} /> End-to-end private legal sessions
          </div>
        </div>
      </div>
    </main>
  )
}

function Message({ message }) {
  const isUser = message.role === 'user'
  return (
    <article className={`message-row ${isUser ? 'user-row' : 'assistant-row'}`}>
      {!isUser && <div className="assistant-avatar"><Scale size={15} /></div>}
      <div className="message-body">
        <div className="message-meta">
          {isUser ? 'You' : 'JurixAI'}
          {!isUser && <span> · Legal AI support</span>}
        </div>
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
          {isUser ? message.content : <FormattedText text={message.content} />}
        </div>
      </div>
      {isUser && <div className="user-avatar">You</div>}
    </article>
  )
}

/* Lightweight safe Markdown renderer for legal answers */
function FormattedText({ text }) {
  const lines = (text || '').split('\n')
  const blocks = []
  let listItems = []
  let listType = null

  const flushList = () => {
    if (!listItems.length) return
    const Tag = listType === 'ol' ? 'ol' : 'ul'
    blocks.push(
      <Tag key={`list-${blocks.length}`} className="md-list">
        {listItems.map((item, idx) => <li key={idx}>{renderInline(item)}</li>)}
      </Tag>
    )
    listItems = []
    listType = null
  }

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim()
    if (!line) {
      flushList()
      return
    }
    const ulMatch = line.match(/^[-*•]\s+(.*)$/)
    const olMatch = line.match(/^\d+\.\s+(.*)$/)
    const hMatch = line.match(/^#{1,3}\s+(.*)$/)

    if (ulMatch) {
      if (listType && listType !== 'ul') flushList()
      listType = 'ul'
      listItems.push(ulMatch[1])
    } else if (olMatch) {
      if (listType && listType !== 'ol') flushList()
      listType = 'ol'
      listItems.push(olMatch[1])
    } else if (hMatch) {
      flushList()
      blocks.push(<h4 key={`h-${index}`} className="md-heading">{renderInline(hMatch[1])}</h4>)
    } else {
      flushList()
      blocks.push(<p key={`p-${index}`} className="md-para">{renderInline(line)}</p>)
    }
  })
  flushList()

  return <div className="md-root">{blocks}</div>
}

function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={i}>{part.slice(2, -2)}</strong>
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={i} className="md-code">{part.slice(1, -1)}</code>
    }
    return part
  })
}

/* Sharp Canva-matched Logo: Crisp geometric Scales + mini Gavel + Plus Jakarta Sans wordmark */
function JurixLogo() {
  return (
    <div className="jurix-brand" aria-label="JurixAI — Legal AI support">
      <svg className="jurix-scales" viewBox="0 0 132 132" fill="none" xmlns="http://www.w3.org/2000/svg">
        {/* Suspension strings */}
        <path
          d="M25 46L7 92M25 46L25 92M25 46L43 92M107 46L89 92M107 46L107 92M107 46L125 92"
          stroke="#FFFFFF"
          strokeWidth="1.6"
          strokeLinecap="round"
        />
        {/* Left & Right crescent pans */}
        <path d="M4 92H46C44 101 35 106 25 106C15 106 6 101 4 92Z" fill="#FFFFFF" />
        <path d="M86 92H128C126 101 117 106 107 106C97 106 88 101 86 92Z" fill="#FFFFFF" />
        {/* Finial, curved beam, pillar, and flared base */}
        <path
          d="M66 6L69.5 18.5C70.5 22 72 24 75 26C86 33 98 37 111 39L109 45.5C96 44 83 39 71 32V106L78 117L90 123V128H42V123L54 117L61 106V32C49 39 36 44 23 45.5L21 39C34 37 46 33 57 26C60 24 61.5 22 62.5 18.5L66 6Z"
          fill="#FFFFFF"
        />
        {/* Subtle pillar collar accents */}
        <rect x="59" y="54" width="14" height="3" rx="1.5" fill="#FFFFFF" />
        <rect x="60" y="96" width="12" height="3" rx="1.5" fill="#FFFFFF" />
      </svg>

      <div className="jurix-wordmark-group">
        <div className="jurix-title-row">
          <svg className="jurix-gavel" viewBox="0 0 42 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            {/* Gavel handle */}
            <path d="M21 15L32 29" stroke="#D9E0FF" strokeWidth="3.2" strokeLinecap="round" />
            {/* Gavel head */}
            <rect x="9" y="7" width="16" height="10" rx="2" transform="rotate(-38 9 7)" fill="#FFFFFF" />
            <rect x="6.5" y="10" width="4" height="11" rx="1.5" transform="rotate(-38 6.5 10)" fill="#C9D3FF" />
            <rect x="19" y="0.5" width="4" height="11" rx="1.5" transform="rotate(-38 19 0.5)" fill="#C9D3FF" />
            {/* Sound block */}
            <rect x="5" y="25" width="17" height="4.2" rx="2.1" fill="#E3E8FF" />
          </svg>
          <span className="jurix-wordmark">JurixAI</span>
        </div>
        <span className="jurix-subtitle">Legal AI support</span>
      </div>
    </div>
  )
}

/* Exact Canva Pill Icons */
function UprightPaperclipIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 7V15.5C15 17.433 13.433 19 11.5 19C9.567 19 8 17.433 8 15.5V6.5C8 4.567 9.567 3 11.5 3C13.433 3 15 4.567 15 6.5V14.5C15 15.605 14.105 16.5 13 16.5C11.895 16.5 11 15.605 11 14.5V7.5" />
    </svg>
  )
}

function FilledMicIcon() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none">
      <rect x="8.5" y="2.5" width="7" height="11.5" rx="3.5" fill="currentColor" />
      <path d="M5.5 11C5.5 14.59 8.41 17.5 12 17.5C15.59 17.5 18.5 14.59 18.5 11" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M12 17.5V21.5M9 21.5H15" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  )
}

function RightArrowIcon() {
  return (
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12H19" />
      <path d="M13 6L19 12L13 18" />
    </svg>
  )
}

export default App
