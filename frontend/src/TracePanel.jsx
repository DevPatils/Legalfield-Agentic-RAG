/**
 * The reasoning trace panel (Architecture.md §9).
 *
 * Sits beside the chat rather than hidden in a collapsed log, because the visible
 * reasoning *is* the demo: which sub-queries ran, what came back and at what rank,
 * why a refinement fired, which strategy it used, and which claims failed verification.
 */

function Hit({ hit, onOpen }) {
  const ranks = []
  if (hit.dense_rank != null) ranks.push(`d${hit.dense_rank}`)
  if (hit.sparse_rank != null) ranks.push(`s${hit.sparse_rank}`)
  return (
    <div className="hit">
      <button className="hit-id cite" onClick={() => onOpen(hit.doc_id, hit.section_id)}>
        §{hit.section_id}
      </button>
      <span className="hit-title">{hit.section_title || '—'}</span>
      <span className="hit-rank">{ranks.join(' ') || 'graph'}</span>
    </div>
  )
}

function Step({ kind, name, headline, detail, children }) {
  return (
    <div className={`step ${kind || ''}`}>
      <div className="step-name">{name}</div>
      {headline && <div className="step-headline">{headline}</div>}
      {detail && <p className="step-detail">{detail}</p>}
      {children}
    </div>
  )
}

export default function TracePanel({ trace, running, onOpenClause }) {
  if (!trace.length && !running) {
    return (
      <div className="empty">
        Ask a question and the agent&rsquo;s reasoning appears here —
        sub-queries, retrieved clauses with their ranks, sufficiency verdicts,
        and any claim that fails verification.
      </div>
    )
  }

  return (
    <>
      {trace.map((step, i) => {
        const s = step.state

        if (step.node === 'planner') {
          return (
            <Step
              key={i}
              name="Planner"
              headline={`${s.query_type || '—'} · ${s.needs_retrieval ? 'retrieving' : 'no retrieval'}`}
              detail={s.planner_reasoning}
            >
              {(s.sub_queries || []).map((q, j) => (
                <div className="hit" key={j}>
                  <span className="hit-title">{q}</span>
                </div>
              ))}
            </Step>
          )
        }

        if (step.node === 'retrieve') {
          const latest = (s.iterations || []).slice(-1)[0] || {}
          const hits = latest.retrieved_chunks || []
          const rewritten = latest.trigger === 'query_rewrite'
          return (
            <Step
              key={i}
              name={rewritten ? 'Retrieve (after rewrite)' : 'Retrieve'}
              headline={`${hits.length} clauses · ${s.context_size ?? 0} in context`}
            >
              {hits.slice(0, 8).map((h) => (
                <Hit key={h.chunk_id} hit={h} onOpen={onOpenClause} />
              ))}
            </Step>
          )
        }

        if (step.node === 'sufficiency') {
          const ok = s.sufficient
          return (
            <Step
              key={i}
              name="Sufficiency check"
              kind={ok ? '' : 'flagged'}
              headline={ok ? 'Context is sufficient' : 'Context is insufficient'}
              detail={ok ? undefined : s.missing_info}
            >
              {!ok && (s.sections_needed || []).length > 0 && (
                <div className="step-detail">
                  Needs sections:{' '}
                  {s.sections_needed.map((x) => (
                    <span className="badge" key={x}>§{x}</span>
                  ))}
                </div>
              )}
              {!ok && (s.terms_needed || []).length > 0 && (
                <div className="step-detail">
                  Needs definitions:{' '}
                  {s.terms_needed.map((x) => (
                    <span className="badge" key={x}>{x}</span>
                  ))}
                </div>
              )}
            </Step>
          )
        }

        if (step.node === 'refine') {
          const latest = (s.iterations || []).slice(-1)[0] || {}
          const traversal = s.refine_strategy === 'graph_traversal'
          return (
            <Step
              key={i}
              name="Refine"
              kind={traversal ? 'traversal' : ''}
              headline={traversal ? 'Graph traversal' : 'Query rewrite'}
              detail={
                traversal
                  ? `Followed stored cross-reference edges to ${(latest.pulled_sections || []).join(', ')} — fetched by id, no second search.`
                  : latest.rationale
              }
            >
              {traversal
                ? (latest.retrieved_chunks || []).map((h) => (
                    <Hit key={h.chunk_id} hit={h} onOpen={onOpenClause} />
                  ))
                : latest.rewritten_query && (
                    <div className="hit">
                      <span className="hit-title">{latest.rewritten_query}</span>
                    </div>
                  )}
            </Step>
          )
        }

        if (step.node === 'flag' && s.low_confidence) {
          return (
            <Step
              key={i}
              name="Confidence"
              kind="flagged"
              headline="Low confidence"
              detail="Answering from context the sufficiency check considered incomplete."
            />
          )
        }

        if (step.node === 'generate') {
          const bad = s.unverified_citations || []
          return (
            <Step
              key={i}
              name="Generate"
              kind={bad.length ? 'flagged' : ''}
              headline={`${(s.citations || []).length} citations verified against context`}
              detail={
                bad.length
                  ? `${bad.length} citation(s) named a clause that was not retrieved: ${bad.join(', ')}`
                  : undefined
              }
            />
          )
        }

        if (step.node === 'faithfulness') {
          const f = s.faithfulness || {}
          const flagged = f.flagged || []
          return (
            <Step
              key={i}
              name="Faithfulness check"
              kind={flagged.length ? 'flagged' : ''}
              headline={
                f.skipped
                  ? 'Skipped — no cited claims'
                  : `${f.claims_checked || 0} claims checked, ${flagged.length} flagged`
              }
              detail={
                flagged.length
                  ? undefined
                  : 'Each claim was entailment-checked against only its own cited clause.'
              }
            >
              {flagged.map((flag, j) => (
                <div className="flag" key={j}>
                  <div className="flag-claim">&ldquo;{flag.claim}&rdquo;</div>
                  <div className="flag-issue">
                    {flag.cited_section ? `[${flag.cited_section}] ` : ''}
                    {flag.issue}
                  </div>
                </div>
              ))}
            </Step>
          )
        }

        return null
      })}

      {running && <Step kind="running" name="Working" headline="…" />}
    </>
  )
}
