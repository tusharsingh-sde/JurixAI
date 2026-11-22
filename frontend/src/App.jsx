import { useEffect, useRef, useState } from 'react'
import {
  Check, FileText, LoaderCircle, LogOut, Mail, Menu,
  MessageCircle, Plus, Scale, ShieldCheck, Sparkles, Trash2, X,
} from 'lucide-react'
import { supabase } from './lib/supabase'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

const HERO_TAGS = [
  'BNS · BNSS · IPC',
  'OCR Notice & PDF Review',
  'Hinglish & English',
]

const QUICK_PROMPTS = [
  {
    label: 'Check FIR rights under BNSS',
    prompt: 'What are my legal rights when filing an FIR or Zero FIR under BNSS in India?',
  },
  {
    label: 'Tenant deposit dispute',
    prompt: 'My landlord is refusing to return my security deposit after vacating. What legal steps can I take in India?',
  },
  {
    label: 'IPC to BNS lookup',
    prompt: 'Explain the key changes from IPC Section 420 (cheating) to the corresponding section in BNS.',
  },
  {
    label: 'Review a legal notice',
    prompt: 'I received a legal notice. What are the key things I should check and how many days do I have to reply?',
  },
]

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
  const [session, setSession] = useState(null)
  const [authReady, setAuthReady] = useState(false)
  const [authOpen, setAuthOpen] = useState(false)
  const [authMode, setAuthMode] = useState('login') // 'login' | 'signup' | 'magic'
  const [oauthBusy, setOauthBusy] = useState('') // 'google' | 'github' | ''
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
  const [guestDoc, setGuestDoc] = useState(null)

  const token = session?.access_token || null
  const email = session?.user?.email || ''
  const displayName =
    session?.user?.user_metadata?.full_name ||
    session?.user?.user_metadata?.user_name ||
    email.split('@')[0] ||
    'User'
  const avatarUrl = session?.user?.user_metadata?.avatar_url || ''
  const authProvider = session?.user?.app_metadata?.provider || 'email'

  const endRef = useRef(null)
  const inputRef = useRef(null)
  const fileRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, busy])

  // Sync Supabase session + handle OAuth redirect return
  useEffect(() => {
    let mounted = true

    supabase.auth.getSession().then(({ data: { session: current } }) => {
      if (!mounted) return
      setSession(current)
      setAuthReady(true)
    })

    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, nextSession) => {
      setSession(nextSession)
      if (event === 'SIGNED_IN' && nextSession) {
        setAuthOpen(false)
        setOauthBusy('')
        setError('')
        setGuestDoc(null)
        setMessages([])
        setActiveSession(null)
      }
    })

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  // Load sessions & flush pending prompt/upload once logged in
  useEffect(() => {
    if (!token) return
    loadSessions(token)

    const savedPrompt = sessionStorage.getItem('jurixai_pending_prompt')
    const savedUpload = sessionStorage.getItem('jurixai_pending_upload')

    if (savedPrompt) {
      sessionStorage.removeItem('jurixai_pending_prompt')
      sendMessage(savedPrompt, token)
    } else if (savedUpload) {
      sessionStorage.removeItem('jurixai_pending_upload')
      setTimeout(() => fileRef.current?.click(), 180)
    }
  }, [token])

  useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        newConversation()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  async function loadSessions(activeToken = token) {
    if (!activeToken) return
    try {
      const data = await request('/sessions', { token: activeToken })
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
    setGuestDoc(null)
    setDraft('')
    setError('')
    setNotice('')
    setSidebarOpen(false)
    setTimeout(() => inputRef.current?.focus(), 60)
  }

  async function authenticate(event) {
    event.preventDefault()
    setError('')
    setNotice('')
    setAuthBusy(true)
    const cleanEmail = authEmail.trim()

    try {
      if (authMode === 'magic') {
        const { error: otpError } = await supabase.auth.signInWithOtp({
          email: cleanEmail,
          options: { emailRedirectTo: window.location.origin },
        })
        if (otpError) throw otpError
        setNotice(`Sign-in link sent to ${cleanEmail}. Check your inbox.`)
        return
      }

      if (authMode === 'signup') {
        const { data, error: signUpError } = await supabase.auth.signUp({
          email: cleanEmail,
          password: authPassword,
          options: { emailRedirectTo: window.location.origin },
        })
        if (signUpError) throw signUpError
        if (!data.session) {
          setNotice('Account created! Verify your email from your inbox, then sign in.')
          setAuthMode('login')
        }
        setAuthPassword('')
        return
      }

      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: cleanEmail,
        password: authPassword,
      })
      if (signInError) throw signInError
      setAuthPassword('')
    } catch (e) {
      setError(e.message)
    } finally {
      setAuthBusy(false)
    }
  }

  async function signInWithProvider(provider) {
    setError('')
    setNotice('')
    setOauthBusy(provider)
    try {
      const { error: oauthError } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: window.location.origin,
          queryParams: provider === 'google' ? { prompt: 'select_account' } : undefined,
        },
      })
      if (oauthError) throw oauthError
    } catch (e) {
      setError(e.message)
      setOauthBusy('')
    }
  }

  async function signOut() {
    try {
      await supabase.auth.signOut()
    } catch { /* ignore */ }
    sessionStorage.removeItem('jurixai_pending_prompt')
    sessionStorage.removeItem('jurixai_pending_upload')
    setSession(null)
    setSessions([])
    setActiveSession(null)
    setMessages([])
    setGuestDoc(null)
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

  async function sendMessage(text = draft, activeToken = token, doc = null) {
    const content = text.trim()
    if (!content || busy) return

    const isGuest = !activeToken
    const recentGuestHistory = isGuest
      ? messages.slice(-6).map((m) => ({
          role: m.role === 'user' ? 'User' : 'JurixAI',
          content: (m.content || '').slice(0, 4000),
        }))
      : null

    const activeDoc = doc || (isGuest ? guestDoc : null)

    setDraft('')
    setError('')
    const userMessage = { id: crypto.randomUUID(), role: 'user', content }
    setMessages((current) => [...current, userMessage])
    setBusy(true)

    try {
      const payload = {
        message: content,
        ...(isGuest
          ? { guest_history: recentGuestHistory }
          : { session_id: activeSession?.id || null }),
        ...(activeDoc?.document_text
          ? { document_text: activeDoc.document_text, document_name: activeDoc.document_name }
          : {}),
      }
      const data = await request('/chat', {
        method: 'POST',
        token: activeToken,
        body: JSON.stringify(payload),
      })
      setMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: 'assistant', content: data.reply },
      ])
      if (!isGuest) {
        if (data.session_id && data.session_id !== activeSession?.id) {
          setActiveSession({
            id: data.session_id,
            title: content.length > 50 ? `${content.slice(0, 50)}...` : content,
          })
        }
        await loadSessions(activeToken)
      }
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
      if (data.truncated) {
        setNotice(`"${file.name}" was large — analyzed the first 30,000 characters.`)
      }
      const docPayload = { document_text: data.extracted_text, document_name: file.name }
      if (!token) setGuestDoc(docPayload)
      setPendingFile(null)
      await sendMessage(
        `Please review the uploaded document "${file.name}" and explain the key details, risks, deadlines, and what I should do next.`,
        token,
        docPayload
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

  if (!authReady) {
    return (
      <main className="cobalt-shell">
        <LoaderCircle className="spin" size={28} />
      </main>
    )
  }

  if (authOpen && !token) {
    return (
      <AuthScreen
        authMode={authMode}
        setAuthMode={(mode) => { setAuthMode(mode); setError(''); setNotice('') }}
        authEmail={authEmail}
        setAuthEmail={setAuthEmail}
        authPassword={authPassword}
        setAuthPassword={setAuthPassword}
        authenticate={authenticate}
        onOAuth={signInWithProvider}
        authBusy={authBusy}
        oauthBusy={oauthBusy}
        error={error}
        notice={notice}
        onBack={() => { setAuthOpen(false); setError(''); setNotice('') }}
      />
    )
  }

  const inChat = messages.length > 0
  const isGuest = !token

  return (
    <div className={`cobalt-shell ${inChat ? 'chat-dark mode-chat' : ''}`}>
      {sidebarOpen && (
        <button className="sidebar-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''} ${inChat ? 'sidebar-chat-dark' : ''}`}>
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

        {isGuest ? (
          <div className="guest-sidebar-card">
            <div className="guest-card-badge">GUEST MODE</div>
            <strong>Chats aren't saved yet</strong>
            <p>
              Try JurixAI freely right now. Sign in with Google, GitHub, or Email to save conversations and sync across devices.
            </p>
            <button
              type="button"
              className="guest-signin-cta"
              onClick={() => { setSidebarOpen(false); setAuthOpen(true) }}
            >
              Sign in to save history
            </button>
          </div>
        ) : (
          <>
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
          </>
        )}

        <div className="sidebar-bottom">
          <div className="privacy-note">
            <ShieldCheck size={15} />
            <span>{isGuest ? 'Private session' : 'Private & securely saved'}</span>
          </div>
          {!isGuest && (
            <div className="profile-row">
              <div className="avatar">
                {avatarUrl ? (
                  <img src={avatarUrl} alt={displayName} className="avatar-img" referrerPolicy="no-referrer" />
                ) : (
                  displayName.charAt(0).toUpperCase()
                )}
              </div>
              <div className="profile-info">
                <strong>{displayName}</strong>
                <span>{authProvider === 'email' ? 'Email account' : `Signed in via ${authProvider}`}</span>
              </div>
              <button className="icon-btn" onClick={signOut} aria-label="Sign out" title="Sign out">
                <LogOut size={16} />
              </button>
            </div>
          )}
        </div>
      </aside>

      <main className={`workspace ${inChat ? 'workspace-chat' : 'workspace-hero'}`}>
        <header className={`topbar ${inChat ? 'topbar-chat' : 'topbar-landing'}`}>
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
            {isGuest ? (
              <>
                {inChat && (
                  <button className="topbar-new-btn" onClick={newConversation}>
                    <Plus size={15} /> New chat
                  </button>
                )}
                <button className="topbar-signin-btn" onClick={() => setAuthOpen(true)}>
                  Sign in
                </button>
              </>
            ) : (
              <>
                <span className="secure-pill">
                  <ShieldCheck size={14} /> Private session
                </span>
                <button className="topbar-new-btn" onClick={newConversation}>
                  <Plus size={15} /> New chat
                </button>
              </>
            )}
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
          <div className="hero-stage hero-stage-authed">
            <div className="hero-content">
              <JurixLogo />
              <h1 className="hero-headline">
                ELITE INDIAN<br />LEGAL<br />INTELLIGENCE
              </h1>
              <p className="hero-tagline">
                Instant, statute-grounded clarity for Indian law — analyze legal notices, FIRs, contracts, and everyday rights in seconds.
              </p>
              <div className="hero-tags">
                {HERO_TAGS.map((tag) => (
                  <span key={tag} className="hero-tag">{tag}</span>
                ))}
              </div>
            </div>

            <img
              className="justice-statue"
              src="/lady-justice.png"
              alt="Blue-lit Lady Justice holding balanced scales"
            />

            <div className="hero-bottom">
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
                <div className="prompt-chips">
                  {QUICK_PROMPTS.map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      className="prompt-chip"
                      onClick={() => sendMessage(item.prompt)}
                    >
                      <Sparkles size={12} />
                      <span>{item.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <div className="hero-micro-footer">
                <span>Grounded in Indian Statutes</span>
                <span>·</span>
                <span>Private Sessions</span>
                <span>·</span>
                <span>For formal court filings, consult an advocate</span>
              </div>
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
              {isGuest && (
                <div className="guest-nudge-bar">
                  <span>Guest mode · This chat won't be saved after refresh.</span>
                  <button type="button" onClick={() => setAuthOpen(true)}>
                    Sign in to save
                  </button>
                </div>
              )}
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
        disabled={busy}
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

function AuthScreen({
  authMode, setAuthMode, authEmail, setAuthEmail,
  authPassword, setAuthPassword, authenticate, onOAuth,
  authBusy, oauthBusy, error, notice, onBack,
}) {
  const isBusy = authBusy || Boolean(oauthBusy)

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
          <h2>
            {authMode === 'login' && 'Welcome back.'}
            {authMode === 'signup' && 'Create your account.'}
            {authMode === 'magic' && 'Passwordless sign in.'}
          </h2>
          <p className="auth-desc">
            {authMode === 'magic'
              ? 'We will email you a one-click sign-in link via Supabase.'
              : 'Continue with your preferred account for private legal sessions.'}
          </p>

          {error && <div className="status-banner status-error"><span>{error}</span></div>}
          {notice && (
            <div className="status-banner status-notice">
              <Check size={14} /><span>{notice}</span>
            </div>
          )}

          <div className="oauth-grid">
            <button
              type="button"
              className="oauth-btn"
              onClick={() => onOAuth('google')}
              disabled={isBusy}
            >
              {oauthBusy === 'google' ? <LoaderCircle className="spin" size={16} /> : <GoogleIcon />}
              <span>Google</span>
            </button>

            <button
              type="button"
              className="oauth-btn"
              onClick={() => onOAuth('github')}
              disabled={isBusy}
            >
              {oauthBusy === 'github' ? <LoaderCircle className="spin" size={16} /> : <GitHubIcon />}
              <span>GitHub</span>
            </button>
          </div>

          <div className="auth-divider"><span>or continue with email</span></div>

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

            {authMode !== 'magic' && (
              <label>
                Password
                <input
                  type="password"
                  autoComplete={authMode === 'login' ? 'current-password' : 'new-password'}
                  placeholder="At least 6 characters"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                  minLength={6}
                  required
                />
              </label>
            )}

            <button className="auth-submit-btn" type="submit" disabled={isBusy}>
              {authBusy ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <>
                  <span>
                    {authMode === 'login' && 'Sign in with Email'}
                    {authMode === 'signup' && 'Create account'}
                    {authMode === 'magic' && 'Send Magic Link'}
                  </span>
                  {authMode === 'magic' ? <Mail size={16} /> : <RightArrowIcon />}
                </>
              )}
            </button>
          </form>

          <div className="auth-links-row">
            <button
              type="button"
              className="auth-link-btn"
              onClick={() => setAuthMode(authMode === 'magic' ? 'login' : 'magic')}
            >
              {authMode === 'magic' ? 'Use password instead' : 'Sign in with Magic Link'}
            </button>

            <button
              type="button"
              className="auth-link-btn"
              onClick={() => setAuthMode(authMode === 'signup' ? 'login' : 'signup')}
            >
              {authMode === 'signup' ? 'Already have an account?' : 'Create an account'}
            </button>
          </div>

          <div className="auth-assurance">
            <ShieldCheck size={14} /> Powered by Supabase Auth · End-to-end private
          </div>
        </div>
      </div>
    </main>
  )
}

function GoogleIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="#EA4335" d="M12 5.04c1.62 0 3.07.56 4.21 1.64l3.15-3.15C17.45 1.77 14.97.8 12 .8 7.42.8 3.47 3.43 1.56 7.26l3.67 2.85C6.1 7.47 8.81 5.04 12 5.04z" />
      <path fill="#4285F4" d="M23.49 12.27c0-.79-.07-1.55-.2-2.27H12v4.51h6.47c-.29 1.48-1.14 2.73-2.4 3.58l3.7 2.87c2.16-1.99 3.72-4.92 3.72-8.69z" />
      <path fill="#FBBC05" d="M5.23 14.11a6.94 6.94 0 0 1 0-4.22L1.56 7.04a11.2 11.2 0 0 0 0 9.92l3.67-2.85z" />
      <path fill="#34A853" d="M12 23.2c3.24 0 5.95-1.07 7.94-2.9l-3.7-2.87c-1.08.72-2.45 1.15-4.24 1.15-3.19 0-5.9-2.15-6.87-5.05l-3.67 2.85C3.47 20.57 7.42 23.2 12 23.2z" />
    </svg>
  )
}

function GitHubIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 1.5C6.2 1.5 1.5 6.2 1.5 12c0 4.64 3.01 8.58 7.18 9.97.53.1.72-.23.72-.51v-1.8c-2.92.63-3.54-1.41-3.54-1.41-.48-1.21-1.17-1.54-1.17-1.54-.95-.65.07-.64.07-.64 1.06.07 1.61 1.09 1.61 1.09.94 1.61 2.46 1.14 3.06.87.09-.68.37-1.14.67-1.41-2.33-.27-4.78-1.17-4.78-5.19 0-1.15.41-2.08 1.08-2.82-.11-.27-.47-1.34.1-2.79 0 0 .88-.28 2.89 1.08a10.04 10.04 0 0 1 5.26 0c2-1.36 2.88-1.08 2.88-1.08.58 1.45.22 2.52.11 2.79.67.74 1.08 1.67 1.08 2.82 0 4.03-2.45 4.92-4.79 5.18.38.33.71.97.71 1.95v2.89c0 .28.19.61.73.51A10.51 10.51 0 0 0 22.5 12c0-5.8-4.7-10.5-10.5-10.5z" />
    </svg>
  )
}

function Message({ message }) {
  const isUser = message.role === 'user'
  if (isUser) {
    return (
      <article className="message-row user-row">
        <div className="message-bubble user-bubble">
          {message.content}
        </div>
      </article>
    )
  }

  return (
    <article className="message-row assistant-row">
      <div className="assistant-avatar"><Scale size={16} /></div>
      <div className="message-body">
        <div className="message-meta">
          <strong>JurixAI</strong>
          <span> · Legal AI Support</span>
        </div>
        <div className="message-bubble assistant-bubble">
          <FormattedText text={message.content} />
        </div>
      </div>
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
    <div className="jurix-brand" aria-label="JurixAI — Legal AI Support">
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
        <span className="jurix-subtitle">Legal AI Support</span>
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
