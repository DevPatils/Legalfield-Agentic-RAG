/**
 * Renders an answer with [doc_id §section_id] tags turned into clickable chips.
 *
 * Tags that the backend could not verify against retrieved context are styled as
 * warnings rather than dropped -- surfacing a bad citation is the point; hiding it
 * would make the answer look cleaner than it is.
 */

// Must match RE_CITATION in backend/app/agent/nodes.py
const CITATION = /\[([A-Za-z0-9_-]+)\s*§\s*([^\]\s]+)\]/g

export default function Answer({ text, unverified = [], onOpenClause }) {
  const bad = new Set(unverified)

  return (
    <>
      {text.split('\n\n').map((para, pi) => {
        const parts = []
        let cursor = 0
        let match
        CITATION.lastIndex = 0

        while ((match = CITATION.exec(para)) !== null) {
          if (match.index > cursor) parts.push(para.slice(cursor, match.index))
          const [tag, docId, sectionId] = match
          const isBad = bad.has(tag)
          parts.push(
            <button
              key={`${pi}-${match.index}`}
              className={`cite${isBad ? ' bad' : ''}`}
              title={isBad ? 'This clause was not in the retrieved context' : `Open ${tag}`}
              onClick={() => !isBad && onOpenClause(docId, sectionId)}
            >
              {docId} §{sectionId}
            </button>
          )
          cursor = match.index + tag.length
        }
        if (cursor < para.length) parts.push(para.slice(cursor))

        return <p key={pi}>{parts}</p>
      })}
    </>
  )
}
