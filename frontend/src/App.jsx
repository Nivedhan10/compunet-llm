import React, { useEffect, useRef, useState } from 'react';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const starter = [{ role: 'assistant', content: 'Hi! I’m TinyTalk, a small character-level Transformer trained from scratch. Ask me about how I work.' }];
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
      setMessages(items => [...items, { role: 'assistant', content: data.reply }]);
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
      <div className="top-actions"><span className={`status ${status}`}><i />{status === 'ready' ? 'Model ready' : status === 'needs_training' ? 'Needs training' : status === 'offline' ? 'API offline' : status === 'error' ? 'Model error' : 'Checking model'}</span><button className="clear" onClick={clearChat}>New chat</button></div>
    </header>
    <section className="intro"><div className="eyebrow"><span /> EDUCATIONAL LANGUAGE MODEL</div><h1>A tiny model.<br /><em>Built from first principles.</em></h1><p>Ask about tokens, attention, or training. TinyTalk was trained from random initialization on a small, original dialogue dataset.</p></section>
    <section className="chat" aria-label="Conversation">
      {messages.map((item, index) => <article className={`message ${item.role}`} key={`${index}-${item.content.slice(0, 12)}`}>
        <div className="avatar">{item.role === 'assistant' ? 'T' : 'Y'}</div>
        <div className="bubble"><div className="speaker">{item.role === 'assistant' ? 'TINYTALK' : 'YOU'}</div><p>{item.content}</p></div>
      </article>)}
      {busy && <article className="message assistant"><div className="avatar">T</div><div className="bubble"><div className="speaker">TINYTALK</div><div className="typing"><i /><i /><i /></div></div></article>}
      {error && <div className="error">{error}</div>}
      {messages.length <= 1 && (
        <div className="suggestions">
          <div className="suggestions-label">SUGGESTED QUESTIONS:</div>
          <div className="chips">
            {SUGGESTIONS.map(prompt => (
              <button key={prompt} className="chip" onClick={() => sendMessage(prompt)}>
                {prompt}
              </button>
            ))}
          </div>
        </div>
      )}
      <div ref={bottom} />
    </section>
    <form className="composer" onSubmit={send}><textarea value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(e); } }} placeholder="Ask TinyTalk something..." maxLength={2000} rows="1" aria-label="Your message" /><button disabled={busy || !text.trim()} aria-label="Send message">↑</button><div className="hint">ENTER TO SEND <span>·</span> SHIFT + ENTER FOR NEW LINE</div></form>
    <footer><span>PYTORCH · FASTAPI · REACT</span><span>SMALL MODEL, CLEAR MECHANICS</span></footer>
  </main>;
}
