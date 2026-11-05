import React, { useEffect, useRef, useState } from 'react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const starter = [{ role: 'assistant', content: 'Hi! I’m TinyTalk, a small character-level Transformer trained from scratch. Ask me about how I work.', source: 'rag_grounded' }];
const SUGGESTIONS = [
  'What is attention?',
  'What is an embedding?',
  'What is weight tying?',
  'Why use Pre-LayerNorm?',
  'How do you work?',
];

export default function App() {
  const [messages, setMessages] = useState(() => {
    try { return JSON.parse(localStorage.getItem('tinytalk-history')) || starter; }
    catch { return starter; }
  });
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('checking');
  const bottom = useRef(null);

  useEffect(() => { localStorage.setItem('tinytalk-history', JSON.stringify(messages)); }, [messages]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, busy]);
  useEffect(() => {
    fetch(`${API}/health`)
      .then(async response => {
        const data = await response.json();
        setStatus(response.ok ? (data.status || 'error') : 'error');
      })
      .catch(() => setStatus('offline'));
  }, []);

  async function sendMessage(rawText) {
    const message = rawText.trim();
    if (!message || busy) return;
    const prior = messages.slice(-12);
    setMessages(items => [...items, { role: 'user', content: message }]);
    setText(''); setBusy(true); setError('');
    try {
      const response = await fetch(`${API}/chat`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, history: prior.filter((item, index) => !(index === 0 && item.role === 'assistant' && item.content.startsWith('Hi! I’m TinyTalk'))) }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'The model could not respond.');
      setMessages(items => [...items, { role: 'assistant', content: data.reply, source: data.source, confidence: data.confidence }]);
    } catch (err) { setError(err.message || 'Could not reach the API. Is the backend running?'); }
    finally { setBusy(false); }
  }

  function send(event) {
    event.preventDefault();
    sendMessage(text);
  }

  function clearChat() { setMessages(starter); setError(''); }

  return <main className="shell">
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-mark">T</span><span>TinyTalk <small>FROM SCRATCH</small></span></a>
      <div className="top-actions">
        <span className={`status ${status}`}>
          <i />
          {status === 'ready' ? 'Model ready' : status === 'needs_training' ? 'Needs training' : status === 'offline' ? 'API offline' : status === 'error' ? 'Model error' : 'Checking model'}
        </span>
        <button className="clear" onClick={clearChat}>New chat</button>
      </div>
    </header>

    <div className="workspace">
      <aside className="intro">
        <div className="eyebrow"><span /> EDUCATIONAL LANGUAGE MODEL</div>
        <h1>A tiny model.<br /><em>Built from first principles.</em></h1>
        <p>Ask about tokens, attention, or training. TinyTalk was trained from random initialization on a small, original dialogue dataset.</p>

        <div className="model-specs">
          <div className="spec-item"><span className="spec-label">PARAMS</span><span className="spec-val">112K</span></div>
          <div className="spec-item"><span className="spec-label">VOCAB</span><span className="spec-val">60</span></div>
          <div className="spec-item"><span className="spec-label">CONTEXT</span><span className="spec-val">128</span></div>
          <div className="spec-item"><span className="spec-label">LAYERS</span><span className="spec-val">2</span></div>
        </div>

        <div className="suggestions">
          <div className="suggestions-label">SUGGESTED QUESTIONS</div>
          <div className="chips">
            {SUGGESTIONS.map(prompt => (
              <button key={prompt} className="chip" onClick={() => sendMessage(prompt)}>
                <span>{prompt}</span>
                <span className="chip-arrow">→</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <section className="chat-container">
        <div className="chat" aria-label="Conversation">
          {messages.map((item, index) => <article className={`message ${item.role}`} key={`${index}-${item.content.slice(0, 12)}`}>
            <div className="avatar">{item.role === 'assistant' ? 'T' : 'Y'}</div>
            <div className="bubble">
              <div className="speaker">
                <span>{item.role === 'assistant' ? 'TINYTALK' : 'YOU'}</span>
                {item.role === 'assistant' && item.source && (
                  <span className={`grounding-badge ${item.source}`}>
                    {item.source === 'rag_grounded' && '⚡ Knowledge Base'}
                    {item.source === 'online_learned' && '🌐 Auto-Learned (Online)'}
                    {item.source === 'user_taught' && '🌱 User Taught'}
                    {item.source === 'needs_teaching' && '💡 Needs Teaching'}
                    {item.source === 'generator' && '🧠 Neural Generator'}
                    {item.confidence ? ` (${Math.round(item.confidence * 100)}%)` : ''}
                  </span>
                )}
              </div>
              <p>{item.content}</p>
            </div>
          </article>)}
          {busy && <article className="message assistant">
            <div className="avatar">T</div>
            <div className="bubble">
              <div className="speaker">TINYTALK</div>
              <div className="typing"><i /><i /><i /></div>
            </div>
          </article>}
          {error && <div className="error">{error}</div>}
          <div ref={bottom} />
        </div>

        <form className="composer" onSubmit={send}>
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(e); } }}
            placeholder="Ask TinyTalk something..."
            maxLength={2000}
            rows="1"
            aria-label="Your message"
          />
          <button disabled={busy || !text.trim()} aria-label="Send message">↑</button>
          <div className="hint">ENTER TO SEND <span>·</span> SHIFT + ENTER FOR NEW LINE</div>
        </form>
      </section>
    </div>

    <footer>
      <span>PYTORCH · FASTAPI · REACT</span>
      <span>SMALL MODEL, CLEAR MECHANICS</span>
    </footer>
  </main>;
}
