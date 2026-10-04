import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp, Check, FileText, Mic,
  LoaderCircle, LogOut, Menu, MessageCircle, Paperclip, Plus, Scale,
  ShieldCheck, Sparkles, Trash2, X,
} from 'lucide-react'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
async function request(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers || {})
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (options.body && !(options.body instanceof FormData)) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${API_URL}${path}`, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`)
  return data
}

function App() {
  const [token, setToken] = useState(() => localStorage.getItem('jurixai_token'))
  const [authOpen, setAuthOpen] = useState(false)
  const [pendingPrompt, setPendingPrompt] = useState('')
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

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])
  useEffect(() => {
    if (token) loadSessions()
  }, [token])

  useEffect(() => {
    if (token && pendingPrompt) {
      const prompt = pendingPrompt
      setPendingPrompt('')
      sendMessage(prompt)
    }
  }, [token])

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
      setMessages((data.messages || []).map((message) => ({
        id: message.id,
        role: message.role?.toLowerCase() === 'user' ? 'user' : 'assistant',
        content: message.content,
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
    inputRef.current?.focus()
  }

  async function authenticate(event) {
    event.preventDefault()
    setError('')
    setAuthBusy(true)
    try {
      const data = await request(authMode === 'login' ? '/login' : '/signup', {
        method: 'POST', body: JSON.stringify({ email: authEmail.trim(), password: authPassword }),
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
    } catch (e) { setError(e.message) }
    finally { setAuthBusy(false) }
  }

  async function signOut() {
    try { await request('/logout', { method: 'POST', token }) } catch { /* Local session is still cleared. */ }
    localStorage.removeItem('jurixai_token')
    localStorage.removeItem('jurixai_email')
    setToken(null); setEmail(''); setSessions([]); setActiveSession(null); setMessages([])
  }

  async function removeSession(event, session) {
    event.stopPropagation()
    try {
      await request(`/sessions/${session.id}`, { method: 'DELETE', token })
      setSessions((current) => current.filter((item) => item.id !== session.id))
      if (activeSession?.id === session.id) newConversation()
    } catch (e) { setError(e.message) }
  }

  async function sendMessage(text = draft) {
    const content = text.trim()
    if (!content || busy) return
    if (!token) {
      setDraft(content)
      setPendingPrompt(content)
      setAuthOpen(true)
      return
    }
    setDraft(''); setError(''); setNotice('')
    const userMessage = { id: crypto.randomUUID(), role: 'user', content }
    setMessages((current) => [...current, userMessage])
    setBusy(true)
    try {
      const data = await request('/chat', {
        method: 'POST', token,
        body: JSON.stringify({ session_id: activeSession?.id || null, message: content }),
      })
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'assistant', content: data.reply }])
      let session = activeSession
      if (data.session_id && data.session_id !== activeSession?.id) {
        session = { id: data.session_id, title: content.length > 50 ? `${content.slice(0, 50)}...` : content }
        setActiveSession(session)
      }
      await loadSessions()
    } catch (e) {
      setError(e.message)
      setMessages((current) => current.filter((message) => message.id !== userMessage.id))
      setDraft(content)
    } finally { setBusy(false) }
  }

  async function analyzeFile(file) {
    if (!file) return
    setPendingFile(file); setUploading(true); setError(''); setNotice('')
    const formData = new FormData(); formData.append('file', file)
    try {
      const data = await request('/upload', { method: 'POST', token, body: formData })
      if (!data.extracted_text?.trim()) throw new Error('No readable text was found. Try a clearer scan or image.')
      setPendingFile(null)
      await sendMessage(`Please review this uploaded ${file.name} and explain the important details, risks, deadlines, and what I should do next.\n\nDOCUMENT TEXT:\n${data.extracted_text}`)
    } catch (e) { setError(e.message); setPendingFile(null) }
    finally { setUploading(false); if (fileRef.current) fileRef.current.value = '' }
  }

  function handleSubmit(event) {
    event.preventDefault()
    sendMessage()
  }

  function startVoiceInput() {
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
    recognition.onerror = () => { setListening(false); setError('Microphone input could not start. Check your browser microphone permission.') }
    recognition.onresult = (event) => setDraft((current) => `${current}${current ? ' ' : ''}${event.results[0][0].transcript}`)
    recognition.start()
  }

  if (!token && !authOpen) return <HeroLanding onAsk={(prompt) => { setDraft(prompt); setPendingPrompt(prompt); setAuthOpen(true) }} onUpload={() => setAuthOpen(true)} />
  if (!token) return <AuthScreen {...{ authMode, setAuthMode, authEmail, setAuthEmail, authPassword, setAuthPassword, authenticate, authBusy, error, notice }} onBack={() => setAuthOpen(false)} />

  return (
    <div className="app-shell">
      {sidebarOpen && <button className="sidebar-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />}
      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="brand"><span className="brand-mark"><Scale size={19} /></span><span>jurix<span className="brand-accent">ai</span></span><button className="mobile-close icon-button" onClick={() => setSidebarOpen(false)} aria-label="Close menu"><X size={18} /></button></div>
        <button className="new-chat" onClick={newConversation}><Plus size={17} /> New conversation <span>⌘ K</span></button>
        <div className="side-label">YOUR CONVERSATIONS <span>{sessions.length}</span></div>
        <div className="session-list">
          {sessions.map((session) => <button key={session.id} className={`session-item ${activeSession?.id === session.id ? 'selected' : ''}`} onClick={() => openSession(session)} title={session.title}>
            <MessageCircle size={15} /><span>{session.title || 'New conversation'}</span><span className="session-delete" onClick={(event) => removeSession(event, session)} role="button" aria-label="Delete conversation"><Trash2 size={14} /></span>
          </button>)}
          {sessions.length === 0 && <p className="empty-sessions">Your conversations will appear here.</p>}
        </div>
        <div className="sidebar-bottom"><div className="privacy-note"><ShieldCheck size={16} /><span>Your conversations are private and securely saved.</span></div><div className="profile"><div className="avatar">{email.charAt(0).toUpperCase() || 'J'}</div><div className="profile-info"><strong>{email.split('@')[0]}</strong><span>Free account</span></div><button className="icon-button" onClick={signOut} aria-label="Sign out" title="Sign out"><LogOut size={17} /></button></div></div>
      </aside>

      <main className="main-panel">
        <header className="topbar"><button className="menu-button icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Menu size={20} /></button><div className="hero-wordmark"><Scale size={28} /><span>JurixAI<small>Legal AI support</small></span></div><div className="topbar-right"><span className="secure-label"><ShieldCheck size={14} /> Private session</span><button className="top-new-button" onClick={newConversation}><Plus size={15} /> New chat</button></div></header>

        <section className={`conversation ${messages.length ? 'conversation-active' : ''}`}>
          {messages.length === 0 ? <div className="welcome-wrap">
            <div className="hero-brand"><JurixLogo /></div>
            <h1><span>ELITE INDIAN</span><span>LEGAL</span><span>INTELLIGENCE</span></h1>
            <img className="justice-art" src="/lady-justice.png" alt="Blue-lit Lady Justice holding balanced scales" />
          </div> : <div className="message-list">{messages.map((message) => <Message key={message.id} message={message} />)}{busy && <div className="assistant-row"><div className="assistant-avatar"><Scale size={16} /></div><div className="typing"><span /><span /><span /></div><small>Thinking through this carefully</small></div>}<div ref={endRef} /></div>}
        </section>

        <div className="composer-zone">
          {error && <div className="inline-alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss"><X size={15} /></button></div>}
          {notice && <div className="inline-notice"><Check size={15} />{notice}<button onClick={() => setNotice('')} aria-label="Dismiss"><X size={15} /></button></div>}
          {pendingFile && <div className="file-chip"><FileText size={15} /><span>{pendingFile.name}</span>{uploading && <LoaderCircle className="spin" size={14} />}<button onClick={() => { setPendingFile(null); setUploading(false) }} aria-label="Remove file"><X size={14} /></button></div>}
          <form className="composer" onSubmit={handleSubmit}><button type="button" className="tool-button attach-button" onClick={() => token ? fileRef.current?.click() : setAuthOpen(true)} disabled={busy || uploading} aria-label="Attach a document"><Paperclip size={21} /></button><textarea ref={inputRef} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); handleSubmit(event) } }} placeholder="Ask anything about Indian law..." rows={1} aria-label="Your message" /><div className="composer-tools"><div><input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg" hidden onChange={(event) => analyzeFile(event.target.files?.[0])} /><button type="button" className={`tool-button mic-button ${listening ? 'listening' : ''}`} onClick={startVoiceInput} aria-label="Voice input"><Mic size={24} /></button></div><button className="send-button" type="submit" disabled={!draft.trim() || busy || uploading} aria-label="Send message">{busy ? <LoaderCircle className="spin" size={17} /> : <ArrowUp size={18} />}</button></div></form>
          <div className="disclaimer"><Sparkles size={12} /> JurixAI can make mistakes. For important matters, consult a qualified lawyer.</div>
        </div>
      </main>
    </div>
  )
}

function Message({ message }) {
  const isUser = message.role === 'user'
  return <article className={`message-row ${isUser ? 'user-row' : 'assistant-message-row'}`}>
    {!isUser && <div className="assistant-avatar"><Scale size={16} /></div>}
    <div className="message-body"><div className="message-meta">{isUser ? 'You' : 'JurixAI'}<span>{isUser ? '' : ' · Legal assistant'}</span></div><div className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>{message.content}</div></div>
    {isUser && <div className="user-avatar">You</div>}
  </article>
}

function JurixLogo() {
  return <svg className="jurix-logo-art" viewBox="0 0 366 154" role="img" aria-label="JurixAI — Legal AI support" xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(-7.7 -8.1) scale(1.1)">
    <g fill="none" stroke="#fff" strokeLinecap="round" strokeLinejoin="round">
      <path d="M32 57v53m0-53 23 53m-23-53-21 53m111-53v53m0-53 21 53m-21-53-23 53" strokeWidth="1.6" />
      <path d="M10 110q22 10 45 0l-4 8q-18 9-37 0zm89 0q22 10 45 0l-4 8q-18 9-37 0z" fill="#fff" strokeWidth="1.8" />
    </g>
    <path d="M74 34 77 23l4 11-1 7 8 5c12 7 24 11 37 12l-2 7c-15-2-28-7-42-15v63l8 12 12 8v6H52v-6l12-8 8-12V50c-13 8-27 13-42 15l-2-7c13-1 25-5 37-12l9-5z" fill="#fff" />
    <path d="M73 73h9m-9 24h9m-9 21h9" stroke="#d5d7ff" strokeWidth="2.4" strokeLinecap="round" />
    </g>
    <g transform="translate(8 0)">
    <path d="M170 55h13v48c0 13-7 20-20 20-7 0-12-2-17-6l7-11c3 2 6 4 10 4 5 0 7-3 7-8z" fill="#fff" />
    <g fill="#dedde3" stroke="#bfc0cd" strokeWidth="1.2" strokeLinejoin="round">
      <path d="m177 34 13-11 5 6-13 11z" />
      <path d="m181 36 7-6 10 13-6 5z" />
      <path d="m176 37 9-7 6 8-9 7z" />
      <path d="m165 47 18-2 2 7-19 2z" />
    </g>
    <text x="188" y="101" fill="#fff" fontFamily="Arial, Helvetica, sans-serif" fontSize="57" fontWeight="700" letterSpacing="-2.2">urixAI</text>
    <text x="188" y="128" fill="#d8d9ff" fontFamily="Arial, Helvetica, sans-serif" fontSize="18" fontWeight="400">Legal AI support</text>
    </g>
  </svg>
}

function HeroLanding({ onAsk, onUpload }) {
  const [query, setQuery] = useState('')
  const startLandingVoice = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) return
    const recognition = new SpeechRecognition()
    recognition.lang = 'en-IN'
    recognition.interimResults = false
    recognition.onresult = (event) => setQuery((current) => `${current}${current ? ' ' : ''}${event.results[0][0].transcript}`)
    recognition.start()
  }
  return <main className="hero-landing"><div className="landing-panel"><div className="hero-brand"><JurixLogo /></div><h1><span>ELITE INDIAN</span><span>LEGAL</span><span>INTELLIGENCE</span></h1><img className="justice-art" src="/lady-justice.png" alt="Blue-lit Lady Justice holding balanced scales" /><form className="landing-search" onSubmit={(event) => { event.preventDefault(); onAsk(query || 'I have a legal question') }}><button type="button" className="landing-tool" aria-label="Attach a document" onClick={onUpload}><Paperclip size={22} /></button><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a legal question..." aria-label="Search legal questions" /><button type="button" className="landing-tool microphone" aria-label="Voice input" onClick={startLandingVoice}><Mic size={23} /></button><button className="landing-send" type="submit" aria-label="Ask JurixAI"><ArrowUp size={22} /></button></form></div></main>
}

function AuthScreen({ authMode, setAuthMode, authEmail, setAuthEmail, authPassword, setAuthPassword, authenticate, authBusy, error, notice, onBack }) {
  return <main className="auth-page"><div className="landing-panel auth-hero"><div className="hero-brand"><JurixLogo /></div><h1><span>ELITE INDIAN</span><span>LEGAL</span><span>INTELLIGENCE</span></h1><img className="justice-art" src="/lady-justice.png" alt="" /><button className="auth-back" onClick={onBack}><X size={16} /> Back to home</button></div><div className="auth-card"><div className="auth-card-brand"><span className="brand-mark"><Scale size={18} /></span> Secure access</div><div className="auth-kicker">YOUR LEGAL SAATHI</div><h2>{authMode === 'login' ? 'Welcome back.' : 'A clearer way forward.'}</h2><p className="auth-description">{authMode === 'login' ? 'Sign in to continue your conversations with JurixAI.' : 'Create an account for thoughtful, law-grounded guidance.'}</p>
    {error && <div className="auth-error">{error}</div>}{notice && <div className="auth-notice"><Check size={15} />{notice}</div>}
    <form onSubmit={authenticate} className="auth-form"><label>Email address<input type="email" autoComplete="email" placeholder="you@example.com" value={authEmail} onChange={(e) => setAuthEmail(e.target.value)} required /></label><label>Password<input type="password" autoComplete={authMode === 'login' ? 'current-password' : 'new-password'} placeholder="At least 6 characters" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} minLength={6} required /></label><button className="auth-submit" disabled={authBusy}>{authBusy ? <LoaderCircle className="spin" size={18} /> : authMode === 'login' ? 'Sign in' : 'Create account'} {!authBusy && <ArrowUp size={17} />}</button></form>
    <p className="auth-switch">{authMode === 'login' ? 'New to JurixAI?' : 'Already have an account?'} <button onClick={() => setAuthMode(authMode === 'login' ? 'signup' : 'login')}>{authMode === 'login' ? 'Create an account' : 'Sign in'}</button></p>
    <div className="auth-assurance"><ShieldCheck size={15} /> Your conversations are private and securely stored.</div></div><div className="auth-footer"><span>© 2026 JurixAI</span><span><Scale size={13} /> Built for clarity, grounded in Indian law</span></div></main>
}

export default App
