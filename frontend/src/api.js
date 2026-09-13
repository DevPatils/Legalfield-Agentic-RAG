/**
 * Backend client.
 *
 * The query path uses SSE rather than a plain POST because the point of the UI is
 * watching the agent reason (Architecture.md §9) -- a spinner followed by an answer
 * would hide the thing this project exists to show.
 */

/**
 * Stream one query. Calls `onNode(node, state)` per completed graph node.
 * Returns an abort function.
 */
export function streamQuery({ query, sessionId, docId, onStart, onNode, onDone, onError }) {
  const params = new URLSearchParams({ query })
  if (sessionId) params.set('session_id', sessionId)
  if (docId) params.set('doc_id', docId)

  const source = new EventSource(`/api/query/stream?${params}`)

  source.addEventListener('start', (e) => onStart?.(JSON.parse(e.data)))
  source.addEventListener('node', (e) => {
    const { node, state } = JSON.parse(e.data)
    onNode?.(node, state)
  })
  source.addEventListener('done', (e) => {
    source.close()
    onDone?.(JSON.parse(e.data))
  })
  source.addEventListener('error', (e) => {
    source.close()
    // A transport failure arrives with no data; a server-sent error carries JSON.
    let detail = 'connection to the backend failed'
    try {
      if (e.data) detail = JSON.parse(e.data).error
    } catch {
      /* keep the default */
    }
    onError?.(detail)
  })

  return () => source.close()
}

async function getJSON(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export const getHealth = () => getJSON('/api/health')
export const listDocuments = () => getJSON('/api/documents')
export const getClause = (docId, sectionId) =>
  getJSON(`/api/documents/${docId}?section_id=${encodeURIComponent(sectionId)}`)
export const getHistory = (sessionId) => getJSON(`/api/sessions/${sessionId}/history`)
export const getTrace = (queryId) => getJSON(`/api/query/${queryId}/trace`)
