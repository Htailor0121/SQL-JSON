import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

// Point to backend automatically based on current host
const API_URL = (process.env.REACT_APP_API_URL)
  || `${window.location.protocol}//${window.location.hostname}:8000`;

const DEFAULTS = {
  db_type: 'mysql',
  host: 'localhost',
  port: 3306,
  user: 'root',
  password: '',
  database: '',
};

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [schema, setSchema] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dbConfig, setDbConfig] = useState(null);
  const [dbForm, setDbForm] = useState(DEFAULTS);
  const [dbError, setDbError] = useState('');
  const chatBoxRef = useRef(null);

  useEffect(() => {
    if (dbConfig) {
      axios.post(`${API_URL}/schema`, { db_config: dbConfig })
        .then(res => {
          if (res.data.error) setDbError(res.data.error);
          else {
            setSchema(res.data);
            setDbError('');
          }
        })
        .catch(() => setDbError('Could not connect to database.'));
    }
  }, [dbConfig]);

  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    setMessages((msgs) => [...msgs, { type: 'user', text: query }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/chat`, {
        query,
        db_config: dbConfig
      });
      setMessages((msgs) => [
        ...msgs,
        { type: 'bot', text: res.data.response, download_url: res.data.download_url }
      ]);
    } catch (err) {
      setMessages((msgs) => [
        ...msgs,
        { type: 'bot', text: 'Error: Could not get response from backend.' }
      ]);
    }
    setQuery('');
    setLoading(false);
  };

  const handleDbFormChange = (e) => {
    const { name, value } = e.target;
    setDbForm(f => ({ ...f, [name]: name === 'port' ? Number(value) : value }));
  };

  const handleDbConnect = async (e) => {
    e.preventDefault();
    setDbError('');
    setSchema(null);
    setDbConfig(null);
    try {
      const res = await axios.post(`${API_URL}/schema`, { db_config: dbForm });
      if (res.data.error) {
        setDbError(res.data.error);
      } else {
        setSchema(res.data);
        setDbConfig(dbForm);
        setDbError('');
      }
    } catch {
      setDbError('Could not connect to database.');
    }
  };

  if (!dbConfig || !schema) {
    return (
      <div className="db-form-container">
        <form className="db-form" onSubmit={handleDbConnect}>
          <h2>Connect to Database</h2>
          <label>
            DB Type
            <select name="db_type" value={dbForm.db_type} onChange={handleDbFormChange}>
              <option value="mysql">MySQL</option>
              <option value="postgresql">PostgreSQL</option>
              <option value="sqlite">SQLite</option>
              <option value="mssql">MSSQL</option>
            </select>
          </label>
          {dbForm.db_type !== 'sqlite' && <>
            <label>
              Host
              <input name="host" value={dbForm.host} onChange={handleDbFormChange} required />
            </label>
            <label>
              Port
              <input name="port" type="number" value={dbForm.port} onChange={handleDbFormChange} required />
            </label>
            <label>
              User
              <input name="user" value={dbForm.user} onChange={handleDbFormChange} required />
            </label>
            <label>
              Password
              <input name="password" type="password" value={dbForm.password} onChange={handleDbFormChange} />
            </label>
            <label>
              Database
              <input name="database" value={dbForm.database} onChange={handleDbFormChange} required />
            </label>
          </>}
          {dbForm.db_type === 'sqlite' && <>
            <label>
              Database File Path
              <input name="database" value={dbForm.database} onChange={handleDbFormChange} required />
            </label>
          </>}
          <button type="submit">Connect</button>
          {dbError && <div className="db-error">{dbError}</div>}
        </form>
      </div>
    );
  }

  return (
    <div className="app-container">
      <button className="sidebar-toggle" onClick={() => setSidebarOpen(v => !v)}>
        {sidebarOpen ? '⏴' : '⏵'}
      </button>
      <div className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <h2>Database Schema</h2>
        {schema ? (
          <div className="schema-list">
            {Object.entries(schema).map(([table, columns]) => (
              <div key={table} className="schema-table">
                <div className="schema-table-name">{table}</div>
                <ul className="schema-columns">
                  {columns.map(col => (
                    <li key={col}>{col}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        ) : (
          <div className="schema-loading">Loading schema...</div>
        )}
      </div>
      <h1>NL-SQL Chat</h1>
      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.type}`}>
            {msg.type === 'bot' && (
              <div className="avatar bot" title="Bot">🤖</div>
            )}
            <div className={`bubble ${msg.type}`}>
              {typeof msg.text === 'string' ? msg.text : <pre>{JSON.stringify(msg.text, null, 2)}</pre>}
              {msg.download_url && (
                <a
                  className="download-btn"
                  href={`${API_URL}${msg.download_url}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  download
                >
                  ⬇️ Download Result JSON
                </a>
              )}
            </div>
            {msg.type === 'user' && (
              <div className="avatar user" title="You">U</div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="avatar bot" title="Bot">🤖</div>
            <div className="bubble bot">Loading...</div>
          </div>
        )}
      </div>
      <form className="input-form" onSubmit={handleSend}>
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Ask a question about your data..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !query.trim()}>Send</button>
      </form>
    </div>
  );
}

export default App; 