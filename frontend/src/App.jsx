import { useCallback, useEffect, useRef, useState } from 'react'
import Answer from './Answer'
import TracePanel from './TracePanel'
import { getClause, getHealth, getHistory, getTrace, streamQuery } from './api'

const EXAMPLES = [
  'How may this agreement be amended or waived?',
  'What counts as Confidential Information?',
  'Can the Joint Governance Committee waive compliance?',
  'How do termination provisions differ across these agreements?',
]

function ClauseDrawer({ clause, onClose, onOpenClause }) {
  if (!clause) return null
  const { loading, error, payload, docId, sectionId } = clause
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="drawer-close" onClick={onClose}>close</button>
        <div className="path">{docId} §{sectionId}</div>
        {loading && <p className="empty">Loading clause…</p>}
        {error && <p className="empty">{error}</p>}
        {payload && (
          <>
            <h3>{payload.section_title || '(untitled)'}</h3>
            {payload.path && <div className="path">{payload.path}</div>}
            <pre>{payload.text}</pre>
            {(payload.cross_references || []).length > 0 && (
              <>
                <div className="pane-title" style={{ marginTop: '1.25rem' }}>
                  Cross-references
                </div>
                <div className="edge-list">
                  {payload.cross_references.map((ref, i) => (
                    <button
                      key={ref + i}
                      className="cite"
                      onClick={() => onOpenClause(payload.doc_id, ref)}
                    >
                      §{ref}
                    </button>
                  ))}
                </div>
              </>
            )}
            {(payload.defined_terms_used || []).length > 0 && (
              <>
                <div className="pane-title" style={{ marginTop: '1rem' }}>
                  Defined terms used
                </div>
                <div className="edge-list">
                  {payload.defined_terms_used.map((t) => (
                    <span className="badge" key={t}>{t}</span>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </aside>
    </div>
  )
}

export default function App() {
  const [sessionId, setSessionId] = useState('')
  const [messages, setMessages] = useState([])
  const [trace, setTrace] = useState([])
  const [running, setRunning] = useState(false)
  const [input, setInput] = useState('')
  const [health, setHealth] = useState(null)
  const [history, setHistory] = useState([])
  const [activeQuery, setActiveQuery] = useState(null)
  const [clause, setClause] = useState(null)
  const abortRef = useRef(null)
  const messagesEnd = useRef(null)
  const traceEnd = useRef(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'down' }))
  }, [])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    traceEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trace])

  const refreshHistory = useCallback((sid) => {
    if (sid) getHistory(sid).then(setHistory).catch(() => {})
  }, [])

  const openClause = useCallback(async (docId, sectionId) => {
    setClause({ loading: true, docId, sectionId })
    try {
      const data = await getClause(docId, sectionId)
      setClause({ docId, sectionId, payload: data.sections[0] })
    } catch {
      setClause({ docId, sectionId, error: `Could not load ${docId} §${sectionId}` })
    }
  }, [])

  const send = useCallback(
    (text) => {
      const question = (text ?? input).trim()
      if (!question || running) return

      setInput('')
      setTrace([])
      setActiveQuery(null)
      setMessages((m) => [...m, { role: 'user', text: question }])
      setRunning(true)

      let final = {}
      abortRef.current = streamQuery({
        query: question,
        sessionId,
        onStart: ({ session_id, query_id }) => {
          if (!sessionId) setSessionId(session_id)
          setActiveQuery(query_id)
        },
        onNode: (node, state) => {
          final = { ...final, ...state }
          setTrace((t) => [...t, { node, state }])
        },
        onDone: ({ query_id }) => {
          setRunning(false)
          setMessages((m) => [
            ...m,
            {
              role: 'agent',
              queryId: query_id,
              summary: final.summary || '',
              text: final.final_answer || '(no answer returned)',
              citations: final.citations || [],
              unverified: final.unverified_citations || [],
              faithfulness: final.faithfulness || {},
              lowConfidence: final.low_confidence,
              confidence: final.confidence,
              iterations: final.iteration,
              tokens: final.token_usage,
            },
          ])
          refreshHistory(sessionId || final.session_id)
        },
        onError: (detail) => {
          setRunning(false)
          setMessages((m) => [...m, { role: 'agent', text: `Request failed: ${detail}`, error: true }])
        },
      })
    },
    [input, running, sessionId, refreshHistory]
  )

  const replay = useCallback(async (queryId) => {
    try {
      const t = await getTrace(queryId)
      setActiveQuery(queryId)
      setMessages([
        { role: 'user', text: t.raw_query },
        {
          role: 'agent',
          queryId,
          summary: t.summary || '',
          text: t.final_answer || '(no answer stored)',
          citations: t.citations || [],
          unverified: t.unverified_citations || [],
          faithfulness: t.faithfulness || {},
          lowConfidence: t.low_confidence,
          confidence: t.confidence,
          iterations: t.iteration,
          tokens: t.token_usage,
          replayed: true,
        },
      ])
      // Reconstruct trace steps from the stored document rather than re-running.
      const steps = []
      if (t.query_type) steps.push({ node: 'planner', state: t })
      ;(t.iterations || []).forEach((it) => {
        steps.push({
          node: it.trigger === 'graph_traversal' ? 'refine' : 'retrieve',
          state: { ...t, iterations: [it], refine_strategy: it.trigger },
        })
        if (it.sufficiency) {
          steps.push({
            node: 'sufficiency',
            state: {
              sufficient: it.sufficiency.sufficient,
              missing_info: it.sufficiency.missing,
              sections_needed: it.sufficiency.sections_needed,
              terms_needed: it.sufficiency.terms_needed,
            },
          })
        }
      })
      if (t.final_answer) steps.push({ node: 'generate', state: t })
      if (t.faithfulness) steps.push({ node: 'faithfulness', state: t })
      setTrace(steps)
    } catch {
      /* ignore */
    }
  }, [])

  const newChat = () => {
    abortRef.current?.()
    setMessages([])
    setTrace([])
    setActiveQuery(null)
    setRunning(false)
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>Agentic RAG — Legal</h1>
          <p>
            <span className={`status-dot ${health?.status === 'ok' ? '' : 'off'}`} />
            {health?.status === 'ok'
              ? `${health.chunks.toLocaleString()} clauses`
              : 'backend offline'}
          </p>
        </div>
        <button className="new-chat" onClick={newChat}>New question</button>
        <div className="pane-head"><span className="pane-title">History</span></div>
        <div className="pane-body">
          {history.length === 0 && <div className="empty">No past questions yet.</div>}
          {history.map((h) => (
            <button
              key={h.query_id}
              className={`history-item${activeQuery === h.query_id ? ' active' : ''}`}
              onClick={() => replay(h.query_id)}
            >
              {h.raw_query}
              <small>replay stored trace</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-pane">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              Ask about the indexed contracts — obligations, definitions,
              termination, payment terms. Every claim comes back cited to a clause.
            </div>
          )}
          {messages.map((m, i) =>
            m.role === 'user' ? (
              <div className="msg" key={i}><div className="msg-user">{m.text}</div></div>
            ) : (
              <div className="msg" key={i}>
                <div className="msg-agent">
                  <Answer
                    summary={m.summary}
                    text={m.text}
                    unverified={m.unverified}
                    onOpenClause={openClause}
                  />
                  {!m.error && (
                    <div className="answer-meta">
                      {m.replayed && <span className="badge">replayed</span>}
                      {m.lowConfidence && <span className="badge warn">low confidence</span>}
                      {m.faithfulness?.flagged?.length > 0 ? (
                        <span className="badge warn">
                          {m.faithfulness.flagged.length} claim(s) flagged
                        </span>
                      ) : (
                        m.faithfulness?.claims_checked > 0 && (
                          <span className="badge ok">
                            {m.faithfulness.claims_checked} claims verified
                          </span>
                        )
                      )}
                      {m.unverified?.length > 0 && (
                        <span className="badge warn">{m.unverified.length} bad citation(s)</span>
                      )}
                      <span>{m.citations?.length || 0} citations</span>
                      {m.iterations != null && <span>{m.iterations} iteration(s)</span>}
                      {m.tokens?.input != null && (
                        <span>{m.tokens.input}in / {m.tokens.output}out tokens</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          )}
          <div ref={messagesEnd} />
        </div>

        <div className="composer">
          <div className="composer-row">
            <textarea
              id="question-input"
              value={input}
              placeholder="Ask about the contracts…"
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send()
                }
              }}
            />
            <button onClick={() => send()} disabled={running || !input.trim()}>
              {running ? 'Working…' : 'Ask'}
            </button>
          </div>
          {messages.length === 0 && (
            <div className="examples">
              {EXAMPLES.map((q) => (
                <button key={q} onClick={() => send(q)}>{q}</button>
              ))}
            </div>
          )}
        </div>
      </main>

      <aside className="trace-pane">
        <div className="pane-head">
          <span className="pane-title">Reasoning trace</span>
          {running && <span className="badge">running</span>}
        </div>
        <div className="pane-body">
          <TracePanel trace={trace} running={running} onOpenClause={openClause} />
          <div ref={traceEnd} />
        </div>
      </aside>

      <ClauseDrawer clause={clause} onClose={() => setClause(null)} onOpenClause={openClause} />
    </div>
  )
}
