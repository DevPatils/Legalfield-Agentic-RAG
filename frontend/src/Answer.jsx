/**
 * Renders an answer: a lead summary, then bullets or prose, with every
 * [doc_id §section_id] tag turned into a clickable chip.
 *
 * Tags that the backend could not verify against retrieved context are styled as
 * warnings rather than dropped -- surfacing a bad citation is the point; hiding it
 * would make the answer look cleaner than it is.
 *
 * The markdown handled here is deliberately only what the generate prompt is allowed
 * to emit: bullet lines and **bold** labels. A full markdown renderer would accept
 * headings, tables and raw HTML that the prompt never asks for, which is surface area
 * for a model's formatting drift to become the UI's problem.
 */

// Must match RE_CITATION in backend/app/agent/nodes.py
const CITATION = /\[([A-Za-z0-9_-]+)\s*§\s*([^\]\s]+)\]/g
const BOLD = /\*\*([^*]+)\*\*/g

/** Split one line into citation chips, bold runs, and plain text. */
function renderInline(line, key, bad, onOpenClause) {
  const parts = []
  let cursor = 0

  // Citations and bold never nest in the generated format, so one combined scan in
  // document order keeps the output faithful to the source ordering.
  const marks = []
  CITATION.lastIndex = 0
  for (let m; (m = CITATION.exec(line)) !== null; ) {
    marks.push({ kind: 'cite', start: m.index, end: m.index + m[0].length, m })
  }
  BOLD.lastIndex = 0
  for (let m; (m = BOLD.exec(line)) !== null; ) {
    marks.push({ kind: 'bold', start: m.index, end: m.index + m[0].length, m })
  }
  marks.sort((a, b) => a.start - b.start)

  for (const mark of marks) {
    if (mark.start < cursor) continue // overlapping match, first one wins
    if (mark.start > cursor) parts.push(line.slice(cursor, mark.start))

    if (mark.kind === 'bold') {
      parts.push(<strong key={`${key}-b${mark.start}`}>{mark.m[1]}</strong>)
    } else {
      const [tag, docId, sectionId] = mark.m
      const isBad = bad.has(tag)
      parts.push(
        <button
          key={`${key}-c${mark.start}`}
          className={`cite${isBad ? ' bad' : ''}`}
          title={isBad ? 'This clause was not in the retrieved context' : `Open ${tag}`}
          onClick={() => !isBad && onOpenClause(docId, sectionId)}
        >
          {docId} §{sectionId}
        </button>
      )
    }
    cursor = mark.end
  }
  if (cursor < line.length) parts.push(line.slice(cursor))
  return parts
}

/** Group lines into bullet runs and paragraphs, preserving order. */
function blocksOf(text) {
  const blocks = []
  let open = null // the block a continuation line would join

  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line) {
      open = null // a blank line ends whatever was being built
      continue
    }

    if (/^[-*•]\s+/.test(line)) {
      const item = line.replace(/^[-*•]\s+/, '')
      if (open?.kind === 'list') open.items.push(item)
      else blocks.push((open = { kind: 'list', items: [item] }))
    } else if (open?.kind === 'list') {
      // A bullet wrapped onto the next line belongs to that bullet, not to a new
      // paragraph -- models wrap long bullets and the indent does not survive here.
      open.items[open.items.length - 1] += ' ' + line
    } else if (open?.kind === 'para') {
      open.text += ' ' + line
    } else {
      blocks.push((open = { kind: 'para', text: line }))
    }
  }
  return blocks
}

export default function Answer({ summary, text, unverified = [], onOpenClause }) {
  const bad = new Set(unverified)
  const inline = (line, key) => renderInline(line, key, bad, onOpenClause)

  return (
    <>
      {summary && <p className="answer-summary">{inline(summary, 'sum')}</p>}

      {blocksOf(text || '').map((block, bi) =>
        block.kind === 'list' ? (
          <ul key={bi} className="answer-list">
            {block.items.map((item, ii) => (
              <li key={ii}>{inline(item, `${bi}-${ii}`)}</li>
            ))}
          </ul>
        ) : (
          <p key={bi}>{inline(block.text, `${bi}`)}</p>
        )
      )}
    </>
  )
}
